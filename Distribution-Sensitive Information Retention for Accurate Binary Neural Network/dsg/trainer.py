"""
basic trainer
"""
import time
import math
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
import paddle

__all__ = ["Trainer"]


class Trainer(object):
    """
    trainer for training network, use SGD
    """
    
    def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
                 train_loader, test_loader, settings, logger, tensorboard_logger=None,
                 opt_type="SGD", optimizer_state=None, run_count=0, mn_perc=0, arc=18):
        """
        init trainer
        """
        
        self.settings = settings
        self.mn_perc = mn_perc 
        self.model = utils.data_parallel(
            model, self.settings.nGPU, self.settings.GPU)
        self.model_teacher = utils.data_parallel(
            model_teacher, self.settings.nGPU, self.settings.GPU)

        self.generator = utils.data_parallel(
            generator, self.settings.nGPU, self.settings.GPU)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.arc = arc
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.bce_logits = nn.BCEWithLogitsLoss().cuda()
        self.MSE_loss = nn.MSELoss().cuda()
        self.lr_master_S = lr_master_S
        self.lr_master_G = lr_master_G
        self.opt_type = opt_type
        import paddle
        if opt_type == "SGD":
            try:
                self.optimizer_S = torch.optim.SGD(
                    params=self.model.parameters(),
                    lr=self.lr_master_S.lr,
                    momentum=self.settings.momentum,
                    weight_decay=self.settings.weightDecay,
                    nesterov=True,
                )
            except:
                self.optimizer_S = paddle.optimizer.SGD(
                    parameters=self.model.parameters(),
                    learning_rate=self.lr_master_S.lr,
                    # momentum=self.settings.momentum,
                    weight_decay=self.settings.weightDecay,
                    # nesterov=True,
                )
        elif opt_type == "RMSProp":
            self.optimizer_S = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum
            )
        elif opt_type == "Adam":
            self.optimizer_S = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay
            )
        else:
            assert False, "invalid type: %d" % opt_type
        if optimizer_state is not None:
            self.optimizer_S.load_state_dict(optimizer_state)\

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
                                            betas=(self.settings.b1, self.settings.b2))

        self.logger = logger
        self.run_count = run_count
        self.scalar_info = {}
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.save_BN_mean = []
        self.save_BN_var = []

        self.fix_G = False
    
    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr_S = self.lr_master_S.get_lr(epoch)
        lr_G = self.lr_master_G.get_lr(epoch)
        # update learning rate of model optimizer
        try:
            for param_group in self.optimizer_S.param_groups:
                param_group['lr'] = lr_S
        except:
            self.optimizer_S.set_lr(lr_S)


        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_G
    
    def loss_fn_kd(self, output, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha

        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """

        criterion_d = nn.CrossEntropyLoss().cuda()
        kdloss = nn.KLDivLoss().cuda()

        alpha = self.settings.alpha
        T = self.settings.temperature
        a = F.log_softmax(output / T, dim=1)
        b = F.softmax(teacher_outputs / T, dim=1)
        c = (alpha * T * T)
        d = criterion_d(output, labels)

        KD_loss = kdloss(a,b)*c + d
        return KD_loss
   
    def compute_gdpp(self, phi_fake, phi_real): 
        def compute_diversity(phi):
            phi = phi.view(phi.size(0), -1)
            phi = F.normalize(phi, p=2, dim=1)
            S_B = torch.mm(phi, phi.t())
            eig_vals, eig_vecs = torch.eig(S_B, eigenvectors=True)
            return Variable(eig_vals[:, 0]), Variable(eig_vecs)

        def normalize_min_max(eig_vals):
            min_v, max_v = torch.min(eig_vals), torch.max(eig_vals)
            return (eig_vals - min_v) / (max_v - min_v)

        fake_eig_vals, fake_eig_vecs = compute_diversity(phi_fake)
        real_eig_vals, real_eig_vecs = compute_diversity(phi_real)
        # Scaling factor to make the two losses operating in comparable ranges.
        magnitude_loss = 0.0001 * F.mse_loss(target=real_eig_vals, input=fake_eig_vals).abs()
        structure_loss = -torch.sum(torch.mul(fake_eig_vecs, real_eig_vecs), 0)
        normalized_real_eig_vals = normalize_min_max(real_eig_vals)
        weighted_structure_loss = torch.sum(torch.mul(normalized_real_eig_vals, structure_loss)).abs()
        return magnitude_loss + weighted_structure_loss

    def forward(self, images, teacher_outputs, labels=None):
        """
        forward propagation
        """
        # forward and backward and optimize
        try:
            output, output_1 = self.model(images, True)
        except:
            output = self.model(paddle.to_tensor(images.cpu().numpy()))
            output = torch.from_numpy(output.cpu().numpy()).cuda()

        if labels is not None:
            loss = self.loss_fn_kd(output, labels, teacher_outputs)
            return output, loss
        else:
            return output, None
    
    def backward_G(self, loss_G):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

    def backward_S(self, loss_S):
        """
        backward propagation
        """
        try:
            self.optimizer_S.zero_grad()
        except:
            self.optimizer_S.clear_grad()

        loss_S.backward()
        self.optimizer_S.step()

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        self.optimizer_S.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()

    def hook_fn_forward(self,module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def hook_fn_forward_saveBN(self,module, input, output):
        self.save_BN_mean.append(module.running_mean.cpu())
        self.save_BN_var.append(module.running_var.cpu())
    
    def train(self, epoch):
        """
        training
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = 200
        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        self.generator.train()
        
        start_time = time.time()
        end_time = start_time
        
        if epoch==0:
            for m in self.model_teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.register_forward_hook(self.hook_fn_forward)
                elif isinstance(m, paddle.nn.BatchNorm2D):
                    m.register_forward_hook(self.hook_fn_forward)
        
        for i in range(iters):
            start_time = time.time()
            data_time = start_time - end_time

            z = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()

            # Get labels ranging from 0 to n_classes for n rows
            labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
            z = z.contiguous()
            labels = labels.contiguous()
            images = self.generator(z, labels)
        
            self.mean_list.clear()
            self.var_list.clear()
            output_teacher_batch, output_teacher_1 = self.model_teacher(images, out_feature = True)

            # One hot loss
            loss_one_hot = self.criterion(output_teacher_batch, labels)

            # BN statistic loss
            BNS_loss = torch.zeros(1).cuda()
            layer_num = 20 
            
            if epoch <= 20:
                self.mn_perc = 30
            elif epoch<=50:
                self.mn_perc = 20
            else:
                self.mn_perc = 10
                
            if self.arc == 18:
                m = np.load('./m_n/m_' + str(self.mn_perc) + '.npy', allow_pickle = True) 
                n = np.load('./m_n/n_' + str(self.mn_perc) + '.npy', allow_pickle = True)   
            else:
                m = np.load('./res50_m_n/m_' + str(self.mn_perc) + '.npy', allow_pickle = True) 
                n = np.load('./res50_m_n/n_' + str(self.mn_perc) + '.npy', allow_pickle = True) 
            for num in range(len(self.mean_list)):
                m_num = m[num] * torch.ones(self.mean_list[num].size()).cuda() 
                n_num = n[num] * torch.ones(self.var_list[num].size()).cuda()
                m_gap = (self.mean_list[num] - self.teacher_running_mean[num]).abs()
                zero = torch.zeros(self.mean_list[num].size()).cuda()
                m_gap = torch.max((m_gap - m_num).abs(), zero)
     
                n_gap = (self.var_list[num] - self.teacher_running_var[num]).abs()
                n_gap = torch.max((n_gap - n_num).abs(), zero)
                m_gap[num] = m_gap[num] * math.sqrt(2)
                n_gap[num] = n_gap[num] * math.sqrt(2)
                m_loss = m_gap.norm() ** 2
                v_loss = n_gap.norm() ** 2
                BNS_loss += m_loss + v_loss
                            #BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
                #       self.var_list[num], self.teacher_running_var[num])

            BNS_loss = BNS_loss / len(self.mean_list)

            gauss_img = torch.randn_like(images).cuda()
            gdpp_loss = self.compute_gdpp(images, gauss_img)
            
            # loss of Generator
            loss_G = loss_one_hot + 0.1 * BNS_loss + 1000.0 * gdpp_loss

            self.backward_G(loss_G)

            output, loss_S = self.forward(images.detach(), output_teacher_batch.detach(), labels)
            
            # if epoch>= self.settings.warmup_epochs:
            #     self.backward_S(loss_S)
            if epoch >= self.settings.warmup_epochs:
                loss_S = paddle.to_tensor(loss_S.cpu().numpy())
                self.backward_S(loss_S)


            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=labels,
                loss=loss_S, top5_flag=True, mean_flag=True)
            
            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))
            
            end_time = time.time()
            
            gt = labels.data.cpu().numpy()
            d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

            fp_acc.update(d_acc)

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f] [gdpp_loss:%f] [S loss: %f] "
        #     % (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item(), gdpp_loss.item(),
        #      loss_S.item())
        # )

        print("[Epoch %d/%d]" % (epoch + 1, self.settings.nEpochs))
        
        self.scalar_info['accuracy every epoch'] = 100 * d_acc
        self.scalar_info['G loss every epoch'] = loss_G
        self.scalar_info['One-hot loss every epoch'] = loss_one_hot
        self.scalar_info['S loss every epoch'] = loss_S

        self.scalar_info['training_top1error'] = top1_error.avg
        self.scalar_info['training_top5error'] = top5_error.avg
        self.scalar_info['training_loss'] = top1_loss.avg
        
        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        return top1_error.avg, top1_loss.avg, top5_error.avg
    
    def test(self, epoch):
        """
        testing
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        
        self.model.eval()
        self.model_teacher.eval()
        
        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            with paddle.no_grad():
                for i, (images, labels) in enumerate(self.test_loader):
                    a = i
                    scale = len(self.test_loader)
                    b = scale - i
                    # c = (i / scale) * 100
                    # dur = time.time() - start_time
                    print('\r' + "[{}->{}]".format(a, b), end="")

                    start_time = time.time()

                    labels = labels.cuda()
                    images = images.cuda()
                    try:
                        output = self.model(images)
                    except:
                        output = self.model(paddle.to_tensor(images.cpu().numpy()))
                        output = torch.from_numpy(output.cpu().numpy()).cuda()

                    loss = torch.ones(1)
                    self.mean_list.clear()
                    self.var_list.clear()

                    single_error, single_loss, single5_error = utils.compute_singlecrop(
                        outputs=output, loss=loss,
                        labels=labels, top5_flag=True, mean_flag=True)

                    top1_error.update(single_error, images.size(0))
                    top1_loss.update(single_loss, images.size(0))
                    top5_error.update(single5_error, images.size(0))

                    end_time = time.time()
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
        )
        # torch.onnx.export(self.model, images, "t.onnx", verbose=True)
        
        self.scalar_info['testing_top1error'] = top1_error.avg
        self.scalar_info['testing_top5error'] = top5_error.avg
        self.scalar_info['testing_loss'] = top1_loss.avg
        if self.tensorboard_logger is not None:
            for tag, value in self.scalar_info.items():
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test_teacher(self, epoch):
        """
        testing
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                data_time = start_time - end_time

                labels = labels.cuda()
                if self.settings.tenCrop:
                    image_size = images.size()
                    images = images.view(
                        image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                    images_tuple = images.split(image_size[0])
                    output = None
                    for img in images_tuple:
                        if self.settings.nGPU == 1:
                            img = img.cuda()
                        img_var = Variable(img, volatile=True)
                        temp_output, _ = self.forward(img_var)
                        if output is None:
                            output = temp_output.data
                        else:
                            output = torch.cat((output, temp_output.data))
                    single_error, single_loss, single5_error = utils.compute_tencrop(
                        outputs=output, labels=labels)
                else:
                    if self.settings.nGPU == 1:
                        images = images.cuda()

                    output = self.model_teacher(images)

                    loss = torch.ones(1)
                    self.mean_list.clear()
                    self.var_list.clear()

                    single_error, single_loss, single5_error = utils.compute_singlecrop(
                        outputs=output, loss=loss,
                        labels=labels, top5_flag=True, mean_flag=True)
                #
                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

        print(
                "Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
                % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
        )

        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg
