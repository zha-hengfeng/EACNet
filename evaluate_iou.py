import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.iouEval import iouEval


def eval(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    iouEvalVal = iouEval(args.classes+1)  # cityscapes
    # iouEvalVal = iouEval(args.classes+1, ignoreIndex=11)    #camvid
    data_list = []
    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = Variable(input).cuda()
            else:
                input_var = Variable(input)
            #label = torch.from_numpy(np.array(label)).long().unsqueeze(0).cuda()
            #label = label.long().unsqueeze(1).cuda()
            start_time = time.time()

            output = model(input_var)

            torch.cuda.synchronize()
            time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        # print(output.max(1)[1].unsqueeze(1).dtype)
        # print(label.dtype)
        iouEvalVal.addBatch(output.max(1)[1].unsqueeze(1).data, label.unsqueeze(1))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        # data_list.append([gt.flatten(), output.flatten()])

        # save the predicted image
        if args.save:
            save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=False)

    #iouVal, iou_classes = iouEvalVal.getIoU()
    #meanIoU, per_class_iu = get_iou(data_list, args.classes)
    meanIoU, per_class_iu = iouEvalVal.getIoU()
    return meanIoU, per_class_iu


def eval_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    logFileLoc = 'log_test_' + args.model + '.txt'
    logFileLoc = os.path.join(os.path.dirname(args.checkpoint), logFileLoc)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)
    #print(model)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    if not args.best:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                print("=====> loading checkpoint '{}'".format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['model'])
                # model.load_state_dict(convert_state_dict(checkpoint['model']))
                # model.load_state_dict(convert_state_dict(checkpoint))
            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

        print("=====> beginning validation")
        print("validation set length: ", len(testLoader))
        mIOU_val, per_class_iu = eval(args, testLoader, model)
        print(mIOU_val)
        print(per_class_iu)

    # Get the best test result among the last 10 model records.
    else:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                dirname, basename = os.path.split(args.checkpoint)
                epoch = int(os.path.splitext(basename)[0].split('_')[1])
                mIOU_val = []
                per_class_iu = []
                min = epoch - args.eval_num + 1
                max = epoch + 1
                for i in range(min, max):
                    basename = 'model_' + str(i) + '.pth'
                    resume = os.path.join(dirname, basename)
                    checkpoint = torch.load(resume)
                    model.load_state_dict(checkpoint['model'])
                    print("=====> beginning test the " + basename)
                    print("validation set length: ", len(testLoader))
                    mIOU_val_0, per_class_iu_0 = eval(args, testLoader, model)
                    mIOU_val.append(mIOU_val_0)
                    per_class_iu.append(per_class_iu_0)
                    logger.write("%d\t%.4f\n" % (i, mIOU_val_0))
                    logger.flush()

                index = list(range(min, max))[np.argmax(mIOU_val)]
                print("The best mIoU among the last 10 models is", index)
                print(mIOU_val)
                per_class_iu = per_class_iu[np.argmax(mIOU_val)]
                mIOU_val = np.max(mIOU_val)
                print(mIOU_val)
                print(per_class_iu)

            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # Save the result
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'log_test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        args.logFile = 'log_test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Mean IoU: %.4f" % mIOU_val)
        logger.write("\nPer class IoU: ")
        for i in range(len(per_class_iu)):
            logger.write("%.4f\t" % per_class_iu[i])
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="EACNet_ResNet-18-ENC", help="model name: Context Guided Network (CGNet)")

    parser.add_argument('--dataset', default="cityscapes", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', default="checkpoint/cityscapes/EACNet_ResNet-18-ENC/bs8_gpu1_train_adam_ohem/model_400.pth")
    parser.add_argument('--eval_num', type=int, default=50)
    # parser.add_argument('--checkpoint', type=str,
    #                     default="./checkpoint/cityscapes/DABNet_cityscapes.pth",
    #                     help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--best', action='store_true', default=True, help="Get the best result among last few checkpoints")
    parser.add_argument('--save', action='store_true', default=False, help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    eval_model(args)
