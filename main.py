import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn

# option file should be modified according to your expriment
from options import Option

from trainer import Trainer
from unet import UNet
from unet.unet_parts import *

import utils as utils
from quantization_utils.quant_modules import *
from utils.data_loading import BasicDataset, CarvanaDataset
from evaluate import evaluate


class Generator(nn.Module):
	# inverse-model generator for U-Net
	def __init__(self, options=None, conf_path=None):
		super(Generator, self).__init__()
		self.settings = options or Option(conf_path)
		self.n_channels = self.settings.channels
		self.bilinear = True

		self.inc = (OutConv(self.settings.nClasses, 64))
		self.down1 = (Down(64, 128))
		self.down2 = (Down(128, 256))
		self.down3 = (Down(256, 512))
		factor = 2 if self.bilinear else 1
		self.down4 = (Down(512, 1024 // factor))
		self.up1 = (Up(1024, 512 // factor, self.bilinear))
		self.up2 = (Up(512, 256 // factor, self.bilinear))
		self.up3 = (Up(256, 128 // factor, self.bilinear))
		self.up4 = (Up(128, 64, self.bilinear))
		self.outc = (DoubleConv(64, self.settings.channels))


	def forward(self, input):
		x1 = self.inc(input)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		img = self.outc(x)
		return img

class ExperimentDesign:
	def __init__(self, generator=None, options=None, conf_path=None):
		self.settings = options or Option(conf_path)
		self.generator = generator
		self.train_loader = None
		self.test_loader = None
		self.model = None
		self.model_teacher = None
		
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0

		self.unfreeze_Flag = True
		
		os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
		os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices
		
		self.settings.set_save_path()
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)

		self.prepare()
	
	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.INFO)
		return logger

	def prepare(self):
		self._set_gpu()
		self._set_dataloader()
		self._set_model()
		self._replace()
		self.logger.info(self.model)
		self._set_trainer()
	
	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		if self.settings.dataset in ["carvana"]:
			try:
				dataset = CarvanaDataset(self.settings.dir_img, self.settings.dir_mask, self.settings.img_scale)
			except (AssertionError, RuntimeError, IndexError):
				dataset = BasicDataset(self.settings.dir_img, self.settings.dir_mask, self.settings.img_scale)
			loader_args = dict(batch_size=self.settings.batchSize, num_workers=os.cpu_count(), pin_memory=True)
			self.train_loader = None
			self.test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

		elif self.settings.dataset in ["nih"]:
			dataset = BasicDataset(self.settings.dir_img, self.settings.dir_mask, self.settings.img_scale)
			loader_args = dict(batch_size=self.settings.batchSize, num_workers=os.cpu_count(), pin_memory=True)
			self.train_loader = None
			self.test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

		else:
			assert False, "unsupport data set: " + self.settings.dataset

	def _set_model(self):
		"""
		Load pre-trained model.
		You can modify this part for your own pre-trained model
		"""

		if self.settings.dataset in ["carvana", "nih"]:
			device = torch.device('cuda')
			self.model = UNet(n_channels=self.settings.channels, n_classes=self.settings.nClasses, bilinear=True)
			self.model_teacher = UNet(n_channels=self.settings.channels, n_classes=self.settings.nClasses, bilinear=True)
			self.model = self.model.to(memory_format=torch.channels_last)
			self.model_teacher = self.model_teacher.to(memory_format=torch.channels_last)
			state_dict = torch.load(self.settings.model_path, map_location=device)
			state_dict_teacher = torch.load(self.settings.model_path, map_location=device)
			del state_dict['mask_values']
			del state_dict_teacher['mask_values']
			self.model.load_state_dict(state_dict)
			self.model_teacher.load_state_dict(state_dict_teacher)
			self.model_teacher.eval()

		else:
			assert False, "unsupport data set: " + self.settings.dataset

	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
		                           self.settings.nEpochs,
		                           self.settings.lrPolicy_S)
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}
		
		lr_master_S.set_params(params_dict=params_dict_S)
		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			generator = self.generator,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=lr_master_S,
			lr_master_G=lr_master_G,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def quantize_model(self,model):
		"""
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
		
		weight_bit = self.settings.qw
		act_bit = self.settings.qa
		
		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		
		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
		
		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model(m))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model(mod))
			return q_model
	
	def _replace(self):
		self.model = self.quantize_model(self.model)
	
	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model

	def run(self):
		# Excute training and test for GDQS
		best_score = 0
		start_time = time.time()

		test_score = evaluate(self.model_teacher, self.test_loader, torch.device('cuda'), True)
		print("Full Precision Model Score: ", test_score)

		try:
			for epoch in range(self.start_epoch, self.settings.nEpochs):
				self.epoch = epoch
				self.start_epoch = 0

				if epoch < 4:
					print ("\n self.unfreeze_model(self.model)\n")
					self.unfreeze_model(self.model)

				train_loss_G, train_loss_S = self.trainer.train(epoch=epoch)

				self.freeze_model(self.model)

				if self.settings.dataset in ["carvana", "nih"]:
					test_score = evaluate(self.model, self.test_loader, torch.device('cuda'), True)
					print("Dice Score is", test_score)
				else:
					assert False, "invalid data set"


				if best_score <= test_score:
					best_score = test_score
				
				print("#==>Best Result (Dice Score) is  ", best_score)
				print("\n")

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return best_score


def main():
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument('--id', type=int, metavar='experiment_id',
	                    help='Experiment ID')
	args = parser.parse_args()
	
	option = Option(args.conf_path)
	option.manualSeed = args.id + 1
	option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)

	if option.dataset in ["carvana", "nih"]:
		generator = Generator(option)
	else:
		assert False, "invalid data set"

	experiment = ExperimentDesign(generator, option)
	experiment.run()


if __name__ == '__main__':
	main()
