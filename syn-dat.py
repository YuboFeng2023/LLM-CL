# encoding=utf-8

import os
import sys
import argparse
import logging
import json
import csv
import random
import importlib
import torch
import texar.torch as tx
from tqdm import tqdm
import numpy as np
import faiss
from copy import deepcopy
import tiktoken
from openai import OpenAI
from jinja2 import Template

from model import EvtRepLrn


def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor",
        type=str,
        required=True,
        help="file path of anchor events")
    parser.add_argument(
        "--atomic",
        type=str,
        required=True,
        help="file path of atomic")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="folder of output")
    parser.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="file path of checkpoint")
    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
        help="file path of prompt")
    parser.add_argument(
        '--config-model',
        type=str,
        default="config_model",
        help="The model config.")
    parser.add_argument(
        '--config-data',
        type=str,
        default="config_data",
        help="The dataset config.")
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="batch size.")
    parser.add_argument(
        '--top-k',
        type=int,
        default=16,
        help="retrieve top k demonstrations.")
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help="openai.")
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help="openai.")
    parser.add_argument(
        '--debug',
        type=int,
        default=1000,
        help="How many example used to be debug.")
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="log file path")

    return parser.parse_args()


def load_anchor_event(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line.strip())
            evt_q = ex['evt_q'].split('\t')
            yield evt_q


def trim(s):
    s = s.replace('_', '')
    l = s.split()
    s = ' '.join(l)
    return s


def others(j, predicate):
    object_list = json.loads(j)
    text_list = list()
    for obj in object_list:
        obj = normalize(obj)

        if obj == 'none':
            continue

        s = 'PersonY %s %s' % (predicate, obj)
        text_list.append(s)
    return text_list


def normalize(s):
    s = s.lower()
    l = s.split()

    for i in range(len(l)):
        if l[i] == 'x':
            l[i] = l[i].replace('x', 'PersonX')
        if l[i] == 'y':
            l[i] = l[i].replace('y', 'PersonY')
    s = ' '.join(l)

    if 'person x' in s:
        s = s.replace('person x', 'PersonX')
    if 'personx' in s:
        s = s.replace('personx', 'PersonX')

    if 'person y' in s:
        s = s.replace('person y', 'PersonY')
    if 'persony' in s:
        s = s.replace('persony', 'PersonY')

    return s


def person_x(j, predicate=None):
    subject_list = json.loads(j)
    text_list = list()
    for subject in subject_list:
        subject = normalize(subject)

        if subject == 'none':
            continue

        if 'PersonX' in subject:
            s = str()
        else:
            s = 'PersonX '

        if predicate:
            s = s + '%s %s' % (predicate, subject)
        else:
            s = s + '%s' % subject

        text_list.append(s)
    return text_list


def load_atomic_event(path):
    with open(path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            yield {
                "event": trim(row['event']),
                "oEffect": others(row['oEffect'], 'will'),
                "oReact": others(row['oReact'], 'will be'),
                "oWant": others(row['oWant'], 'want'),
                "xAttr": person_x(row['xAttr'], 'is'),
                "xEffect": person_x(row['xEffect'], ),
                "xIntent": person_x(row['xIntent'], 'wants'),
                "xNeed": person_x(row['xNeed'], 'needs'),
                "xReact": person_x(row['xReact'], 'will feel'),
                "xWant": person_x(row['xWant'], 'wants'),
            }


def validate(map_list, key, n=10):
    l = list()
    for d in map_list:
        if len(d[key]):
            l.append(d[key])
    l = random.choices(l, k=n)
    return l


def convert_map_to_pair(map_list):
    key_set, pair_list = set(), list()
    for mapping in map_list:
        head = mapping['event']
        for k in ['oEffect', 'oReact', 'oWant', 'xAttr', 'xIntent', 'xNeed', 'xReact', 'xWant']:
            for tail in mapping[k]:
                key = "%s - %s" % (head, tail)
                if key not in key_set:
                    pair_list.append([head, tail])
                key_set.add(key)
    return pair_list


def encode(model, text_list, batch_size=8, device='cpu'):
    tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
    total_batch = len(text_list) // batch_size + (1 if len(text_list) % batch_size > 0 else 0)
    embedding_list = list()
    with torch.no_grad():
        for batch_id in tqdm(range(total_batch)):
            beg, end = batch_id * batch_size, (batch_id + 1) * batch_size
            samples = list()
            for event in text_list[beg: end]:
                event = ' '.join([tokenizer.cls_token] + [event] + [tokenizer.sep_token])
                samples.append(event)
            samples = [tokenizer.map_text_to_id(sample) for sample in samples]
            pad_value = tokenizer.map_token_to_id(tokenizer.pad_token)
            ids, lengths = tx.data.padded_batch(samples, pad_value=pad_value)
            ids = torch.from_numpy(ids).to(device)
            lengths = torch.tensor(lengths).to(device)
            embeddings = model.encoder_q(ids, lengths)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize_to_unit
            embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, dim=0)
    return embeddings.numpy()


def retrieve(
        cdd_embeds: np.ndarray,
        qry_embeds: np.ndarray,
        cdd_events: list,
        qry_events: list,
        top_k=8):

    # build index
    index = faiss.IndexFlatIP(cdd_embeds.shape[1])
    index.add(cdd_embeds.astype(np.float32))
    # search index
    dist_list, idx_list = index.search(qry_embeds.astype(np.float32), top_k)
    # pack
    demos_list, dists_list = list(), list()
    for i, query in enumerate(qry_events):
        demos, dists = list(), list()
        for dist, idx in zip(dist_list[i], idx_list[i]):
            cdd = deepcopy(cdd_events[idx])
            demos.append(cdd)
            dists.append(float(dist))
        demos_list.append(demos)
        dists_list.append(dists)
    return demos_list, dists_list


def tokenizing(s):
    tokenizer = tiktoken.encoding_for_model('text-davinci-003')
    token_list = tokenizer.encode(s)
    return token_list


def event_to_text(evt):
    text = ' '.join(evt)
    return text


def check_event(evt):
    text = event_to_text(evt)
    tokens = tokenizing(text)
    if 3 <= len(tokens) <= 8:
        return True
    else:
        return False


def save_distance(path, evt_q_list, dists_list):
    l = list()
    for evt_q, dists in zip(evt_q_list, dists_list):
        if check_event(evt_q):
            l.append(dists)

    with open(path, 'w', encoding='utf-8') as fo:
        json.dump(obj=l, fp=fo, indent=4)


def synthetize(
        path: str,
        api_key: str,
        base_url: str,
        evt_q: list[str],
        demos: list[str],):

    with open(path, 'r', encoding='utf-8') as f:
        template = f.read()
    template = Template(template)
    prompt = template.render(demos=demos, query=event_to_text(evt_q))
    print(prompt)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=8,
        n=3,
        seed=42,
        temperature=0.7,
        stop=['\n\n'],
    )
    evt_k = [choice.text.strip() for choice in response.choices]

    return evt_k


def run(argv):
    logging.info("Loading Anchor Events ...")
    evt_q_list = [evt_q for evt_q in load_anchor_event(argv.anchor)]

    l = list(filter(check_event, evt_q_list))
    logging.info("#(anchor event) = %d" % len(evt_q_list))
    logging.info("#(standard anchor event) = %d" % len(l))

    logging.info("Loading ATOMIC Events ...")
    atomic_mapping_list = [mapping for mapping in load_atomic_event(argv.atomic)]

    # validation
    for k in ['event', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xIntent', 'xNeed', 'xReact', 'xWant']:
        print("%s: %s" % (k, validate(atomic_mapping_list, k)))

    atomic_pair_list = convert_map_to_pair(atomic_mapping_list)
    logging.info('There are %d ATOMIC mapping' % len(atomic_mapping_list))
    logging.info('There are %d ATOMIC pair' % len(atomic_pair_list))

    # debug
    evt_q_list = random.choices(evt_q_list, k=argv.debug) if argv.debug else evt_q_list
    atomic_pair_list = random.choices(atomic_pair_list, k=argv.debug) if argv.debug else atomic_pair_list

    logging.info('Loading Checkpoint ...')
    config_model = importlib.import_module(argv.config_model)
    config_data = importlib.import_module(argv.config_data)
    model = EvtRepLrn(config_model=config_model, config_data=config_data)
    if argv.ckpt:
        state_dict = torch.load(f=argv.ckpt, map_location='cpu')
        model.load_state_dict(state_dict['model'])

    logging.info("Encoding Anchor Events ...")
    text_list = [event_to_text(evt) for evt in evt_q_list]
    anchor_embeds = encode(model, text_list, argv.batch_size,)

    logging.info("Encoding ATOMIC Events ...")
    head_text_list = [pair[0] for pair in atomic_pair_list]
    tail_text_list = [pair[1] for pair in atomic_pair_list]
    head_embeds = encode(model, head_text_list, argv.batch_size,)
    tail_embeds = encode(model, tail_text_list, argv.batch_size,)
    pair_embeds = (head_embeds + tail_embeds) /2

    logging.info("Retrieve ATOMIC Events by Anchor Events ...")
    demos_list, dists_list = retrieve(pair_embeds, anchor_embeds, atomic_pair_list, evt_q_list, argv.top_k)

    save_distance(os.path.join(argv.output, 'distances_list.json'), evt_q_list, dists_list)

    ds = list()
    for evt_q, demos, dists in zip(evt_q_list, demos_list, dists_list):
        evt_k = list()
        if np.mean(dists) >= 0.7:
            evt_k = synthetize(
                path=argv.prompt,
                api_key=argv.api_key,
                base_url=argv.base_url,
                evt_q=evt_q,
                demos=demos,)
        ds.append({'evt_q': evt_q, 'evt_k': evt_k,})

    with open(os.path.join(argv.output, 'train.json'), 'w', encoding='utf-8') as fo:
        for d in ds:
            fo.write("%s\n" % json.dumps(d))


def main():
    argv = parse_argv()

    for k, v in argv.__dict__.items():
        print("%s: %s" % (k, v))

    logging.basicConfig(
        filename=argv.log,
        format='%(asctime)s - %(levelname)s - %(name)s - %(msg)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    run(argv)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()
