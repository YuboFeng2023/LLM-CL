max_train_epoch = 2
display_steps = 100
eval_steps = 1000
save_steps = 1000
vocab_file=""

train_hparams={
        'dataset': {
                'files': '/data/train.json',
                'vocab_file': vocab_file,
        },
        'batch_size': 256,
        'lazy_strategy': 'all',
        'num_parallel_calls': 1,
        'shuffle_buffer_size': 50000,
        "allow_smaller_final_batch": False,
        "cache_strategy": "none",
        'shuffle': True,
}

valid_hparams={
        'dataset': {
                'files': '/data/valid.json',
                'vocab_file': vocab_file,
        },
        'batch_size': 64,
        "allow_smaller_final_batch": False,
        'shuffle': False,
}

hard_hparams={
        'dataset': {
                'files': '/data/hard.txt',
                'vocab_file': vocab_file,
        },
        'batch_size': 128,
        'shuffle': False
}

hardext_hparams={
        'dataset': {
                'files': '/data/extend.txt',
                'vocab_file': vocab_file,
        },
        'batch_size': 128,
        'shuffle': False
}

trans_hparams={
        'dataset': {
                'files': '/data/trans.txt',
                'vocab_file': vocab_file,
        },
        'batch_size': 128,
        'shuffle': False
}
