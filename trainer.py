"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from evaluate import evaluate
from utils.dice_score import dice_loss
import pyfastnoisesimd as fns

__all__ = ["Trainer"]

class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
	             train_loader, test_loader, settings, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		
		self.model = utils.data_parallel(
			model, self.settings.nGPU, self.settings.GPU)
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		self.generator = utils.data_parallel(
			generator, self.settings.nGPU, self.settings.GPU)

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		self.criterion = nn.CrossEntropyLoss().cuda() if self.model.n_classes > 1 else nn.BCEWithLogitsLoss().cuda()
		self.bce_logits = nn.BCEWithLogitsLoss().cuda()
		self.MSE_loss = nn.MSELoss().cuda()
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type
		if opt_type == "SGD":
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
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
		for param_group in self.optimizer_S.param_groups:
			param_group['lr'] = lr_S

		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr_G
	
	def loss_fn_kd(self, output, labels, teacher_outputs, n_classes):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities!

		The final return is combined with CE Loss and Dice Loss.
		"""

		criterion_d = nn.CrossEntropyLoss().cuda() if n_classes > 1 else nn.BCEWithLogitsLoss().cuda()
		kdloss = nn.KLDivLoss().cuda()

		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, dim=1)
		b = F.softmax(teacher_outputs / T, dim=1)
		c = (alpha * T * T)

		if n_classes == 1:
			loss = criterion_d(output.squeeze(1), labels.float())
			loss += dice_loss(F.sigmoid(output.squeeze(1)), labels.float(), multiclass=False)
		else:
			loss = criterion_d(output, labels)
			loss += dice_loss(
				F.softmax(output, dim=1).float(),
				F.one_hot(labels, n_classes).permute(0, 3, 1, 2).float(),
				multiclass=True
			)

		KD_loss = kdloss(a,b)*c + loss
		return KD_loss
	
	def forward(self, images, teacher_outputs, labels=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize
		output = self.model(images)
		if labels is not None:
			loss = self.loss_fn_kd(output, labels, teacher_outputs, self.model.n_classes)
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
		self.optimizer_S.zero_grad()
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

		iters = 200
		self.update_lr(epoch)

		self.model.eval()
		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time

		shape = [self.settings.img_size, self.settings.img_size]
		seed = np.random.randint(2**31)
		N_threads = 4
		
		if epoch==0:
			for m in self.model_teacher.modules():
				if isinstance(m, nn.BatchNorm2d):
					m.register_forward_hook(self.hook_fn_forward)
		
		for i in range(iters):
			start_time = time.time()
			data_time = start_time - end_time

			# sample full-size 2-channel Gaussian Noise
			z = torch.randn(self.settings.batchSize, 2, self.settings.img_size, self.settings.img_size)

			CF = fns.Noise(seed=seed, numWorkers=N_threads)  # Cubic Fractal Noise
			masks = []
			input = []
			for _ in range(self.settings.batchSize):
				CF.frequency = 0.012 + torch.randn(1) * 0.002  # sample frequency from N(0.012, 0.002^2)
				CF.noiseType = fns.NoiseType.CubicFractal
				CF.fractal.fractalType = fns.FractalType.Billow
				CF.fractal.octaves = 2				
				cf_noise = CF.genAsGrid(shape)  # sample Cubic Fractal Noise
				cf_noise = cf_noise.tolist()

				# generate mask from Cubic Fractal Noise
				mask = np.array(list(map(lambda x: list(map(lambda y: y >= -0.75, x)), cf_noise)))
				mask_reverse = np.array(list(map(lambda x: list(map(lambda y: y < -0.75, x)), cf_noise)))

				# combine mask with Gaussian Noise z
				noise_mask = np.zeros((2, self.settings.img_size, self.settings.img_size)).tolist()
				noise_mask[0] = (torch.max(z[_], dim=0)[0]*mask_reverse + torch.min(z[_], dim=0)[0]*mask).tolist()
				noise_mask[1] = (torch.max(z[_], dim=0)[0]*mask + torch.min(z[_], dim=0)[0]*mask_reverse).tolist()

				mask = mask.tolist()
				input.append(noise_mask)
				masks.append(mask)
			input = torch.Tensor(input).cuda()  # input of generator
			masks = torch.Tensor(masks).cuda()  # generated masks
			input = input.view([self.settings.batchSize, 2, self.settings.img_size, self.settings.img_size])
			
			input = input.contiguous()
			images = self.generator(input)  # generated fake images
		
			self.mean_list.clear()
			self.var_list.clear()
			output_teacher_batch = self.model_teacher(images)

			masks = masks.to(torch.long)

			if self.model.n_classes == 1:
				ce_loss = self.criterion(output_teacher_batch.squeeze(1), masks.float())
				dc_loss = dice_loss(F.sigmoid(output_teacher_batch.squeeze(1)), masks.float(), multiclass=False)
			else:
				ce_loss = self.criterion(output_teacher_batch, masks)
				dc_loss = dice_loss(
					F.softmax(output_teacher_batch, dim=1).float(),
					F.one_hot(masks, self.model.n_classes).permute(0, 3, 1, 2).float(),
					multiclass=True
				)
			loss = ce_loss + dc_loss

			# BN statistic loss
			BNS_loss = torch.zeros(1).cuda()

			for num in range(len(self.mean_list)):
				BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
					self.var_list[num], self.teacher_running_var[num])

			BNS_loss = BNS_loss / (len(self.mean_list))

			# loss of Generator
			loss_G = loss + 0.01 * BNS_loss  # CE Loss & Dice Loss & BNS Loss

			self.backward_G(loss_G)

			output, loss_S = self.forward(images.detach(), output_teacher_batch.detach(), masks)
			
			if epoch>= self.settings.warmup_epochs:
				self.backward_S(loss_S)

			end_time = time.time()

			dc_loss = dice_loss(
						F.softmax(output_teacher_batch, dim=1).float(),
						F.one_hot(masks, self.model.n_classes).permute(0, 3, 1, 2).float(),
						multiclass=True
					)
		print(
			"[Epoch %d/%d] [Batch %d/%d] [G loss: %.4f] [Dice loss: %f] [One-hot loss: %f] [BNS_loss:%f] [S loss: %f] "
			% (epoch + 1, self.settings.nEpochs, i+1, iters, loss_G.item(), dc_loss.item(), ce_loss.item(), BNS_loss.item(),
			 loss_S.item())
		)

		self.scalar_info['G loss every epoch'] = loss_G
		self.scalar_info['Dice loss every epoch'] = dc_loss
		self.scalar_info['One-hot loss every epoch'] = ce_loss
		self.scalar_info['BNS loss every epoch'] = BNS_loss
		self.scalar_info['S loss every epoch'] = loss_S
		
		if self.tensorboard_logger is not None:
			for tag, value in list(self.scalar_info.items()):
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}

		return loss_G, loss_S
	