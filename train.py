import numpy as np
import argparse
import os
import time
import datetime
import csv
from pathlib import Path
from model import FEAT
import tasks
import datautils
from utils import init_dl_program, pkl_save, data_dropout, yaml_save, yaml_load


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
            print(f'------------------------> Saved valid model {n}.')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('--dataset', help='The dataset name')
    parser.add_argument('--sub_dataset', type=str, default=None, help='The sub dataset name')
    parser.add_argument('--run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--result_name', type=str, default='base', help='The file name of result csv.')

    # ts2vec configs
    parser.add_argument('--depth', type=int, default=10, help='The number of hidden residual blocks in the encoder (default 10)')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set toUEA, forecast_csv.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--hidden_dims', type=int, default=64, help='The encoder hidden dimension (defaults to 64)')
    parser.add_argument('--max_train_length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--iters_factor', type=float, default=None, help='The multiplying factor for adjusting default iteration setting.')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save_every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max_threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')

    # FEAT configs
    parser.add_argument('--max_shift_size', type=int, default=None, help='The number of maximum shift size (defaults to 3)')
    parser.add_argument('--recon_loss', type=str, default='l2', help='The reconstruction loss function.')
    parser.add_argument('--temporal_mask_mode', type=str, default='binomial', help='The temporal embedding mask mode.')
    parser.add_argument('--feature_mask_mode', type=str, default='all_true', help='The feature embedding mask mode.')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    print('\n')

    print('-------------- Model Configuration --------------')
    print(f'Max shift size : {args.max_shift_size}')
    print(f'Temp mask mode : {args.temporal_mask_mode}')
    print(f'Feat mask mode : {args.feature_mask_mode}')
    print('\n')

    # make directory to save arguments yaml file
    os.makedirs(f'result/{args.run_name}/arguments', exist_ok=True)

    # save arguments yaml file
    if not Path(f'result/{args.run_name}/arguments/arguments_{args.result_name}.yaml').is_file():
        yaml_save(f'result/{args.run_name}/arguments/arguments_{args.result_name}.yaml', {f'{args.dataset}':vars(args)})
    else:
        args_yaml = yaml_load(f'result/{args.run_name}/arguments/arguments_{args.result_name}.yaml')
        args_yaml[f'{args.dataset}'] = vars(args)
        yaml_save(f'result/{args.run_name}/arguments/arguments_{args.result_name}.yaml', args_yaml)

    # device setting
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    # load dataset
    print('Loading data... ', end='')
    if args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)

    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]

    elif args.loader == 'anomaly_detector':
        task_type = 'anomaly_detection_decoder'
        if args.sub_dataset:
            args.dataset = f'{args.dataset}_{args.sub_dataset}'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = np.expand_dims(all_train_data, axis=0)
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')


    run_dir = f'training/' + f'{args.run_name}/{args.result_name}/' + args.dataset
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    print('Build Model... ', end='')
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        hidden_dims=args.hidden_dims
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    model = FEAT(
        input_dims=train_data.shape[-1],
        device=device,
        context_l=min(train_data.shape[1], args.max_train_length),
        iters_factor=args.iters_factor,
        max_shift_size=args.max_shift_size,
        recon_loss=args.recon_loss,
        temporal_mask_mode=args.temporal_mask_mode,
        feature_mask_mode=args.feature_mask_mode,
        **config
    )
    print('done\n')

    # model description
    num_params = sum(p.numel() for p in model._net.parameters() if p.requires_grad)
    print(f"--> Model Parameters : {num_params/1000000} M")


    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm', max_train_length=args.max_train_length)
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, max_train_length=args.max_train_length)
        else:
            assert False

        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

        result_file = f'result/{args.run_name}/{args.result_name}.csv'

        if not Path(result_file).is_file():
            os.makedirs(f'result/{args.run_name}', exist_ok=True)
            with open(result_file, 'w') as f:
                wr = csv.writer(f)
                wr.writerow(['dataset'] + list(eval_res.keys()) + ['num_params'] + ['training_time'])

        with open(result_file, 'a') as f:
            wr = csv.writer(f)
            wr.writerow([f'{args.dataset}'] +["%.4f" % e for e in list(eval_res.values())] + [f'{num_params}'] + [f'{datetime.timedelta(seconds=t)}'])


    print("Finished.")