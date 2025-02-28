from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
import torch


class Transformer_wrapper(Transformer.Model):
    def __init__(self, configs, flat_shapes=0, orig_shapes=0):
        self.flat_shapes = flat_shapes
        self.orig_shapes = orig_shapes
        super(Transformer_wrapper, self).__init__(configs)
        
    def forward(self, input):
        if not torch.is_tensor(input):
            input = torch.from_numpy(input).to(exp.device)
 
        unflat_batch_x, unflat_batch_x_mark, unflat_dec_inp, unflat_batch_y_mark = input.split(
            [self.flat_shapes[0], self.flat_shapes[1], self.flat_shapes[2], self.flat_shapes[3]],dim=1)

        unflat_batch_x = unflat_batch_x.unflatten(1,self.orig_shapes[0])
        unflat_batch_x_mark = unflat_batch_x_mark.unflatten(1,self.orig_shapes[1])
        unflat_dec_inp = unflat_dec_inp.unflatten(1,self.orig_shapes[2])
        unflat_batch_y_mark = unflat_batch_y_mark.unflatten(1,self.orig_shapes[3])
        return super(Transformer_wrapper, self).forward(unflat_batch_x.float(), unflat_batch_x_mark.float(), unflat_dec_inp.float(), unflat_batch_y_mark.float())

    def set_flat_shapes(self, flat_shapes):
        self.flat_shapes = flat_shapes

    def set_orig_shapes(self, orig_shapes):
        self.orig_shapes = orig_shapes


class Reduced_transformer_wrapper(Transformer.Model):
    def __init__(self, configs, global_batch_x_mark=0, global_batch_y_mark=0, device="cpu"):
        self.global_batch_x_mark = global_batch_x_mark
        self.global_batch_y_mark = global_batch_y_mark
        self.device = device
        self.c_out = configs.c_out
        self.label_len = configs.label_len
        super(Reduced_transformer_wrapper, self).__init__(configs)
        
    def forward(self, input):
        if not torch.is_tensor(input):
            input = torch.from_numpy(input).to(self.device)
        
        unflat_batch_x = input.unflatten(1,[self.pred_len, self.c_out])
        current_dec_inp = torch.zeros((self.global_batch_y_mark.shape[0],self.global_batch_y_mark.shape[1],unflat_batch_x.shape[-1]), dtype=unflat_batch_x.dtype, device=unflat_batch_x.device)
        if (current_dec_inp.shape[0]==unflat_batch_x.shape[0]):
            current_batch_x_mark = self.global_batch_x_mark
            current_batch_y_mark = self.global_batch_y_mark
        else:
            current_dec_inp = current_dec_inp.repeat(unflat_batch_x.shape[0],1,1)
            current_batch_x_mark = self.global_batch_x_mark.repeat(unflat_batch_x.shape[0],1,1)
            current_batch_y_mark = self.global_batch_y_mark.repeat(unflat_batch_x.shape[0],1,1)
            
        current_dec_inp[:,:self.label_len,:] = unflat_batch_x[:,self.label_len:,:]
                
        return super(Reduced_transformer_wrapper, self).forward(unflat_batch_x.float(), current_batch_x_mark.float(), current_dec_inp.float(), current_batch_y_mark.float())

    def set_global_batch_x_mark(self, global_batch_x_mark):
        self.global_batch_x_mark = global_batch_x_mark

    def set_global_batch_y_mark(self, global_batch_y_mark):
        self.global_batch_y_mark = global_batch_y_mark


class Reduced_io_transformer_wrapper(Transformer.Model):
    def __init__(self, configs, global_batch_x_mark=0, global_batch_y_mark=0, device="cpu", pred_hor_explained=0):
        self.pred_time_step = pred_hor_explained
        self.global_batch_x_mark = global_batch_x_mark
        self.global_batch_y_mark = global_batch_y_mark
        self.device = device
        self.c_out = configs.c_out
        self.label_len = configs.label_len
        super(Reduced_io_transformer_wrapper, self).__init__(configs)
        
    def forward(self, input):
        if not torch.is_tensor(input):
            input = torch.from_numpy(input).to(self.device)
        
        unflat_batch_x = input.unflatten(1,[self.pred_len, self.c_out])
        current_dec_inp = torch.zeros((self.global_batch_y_mark.shape[0],self.global_batch_y_mark.shape[1],unflat_batch_x.shape[-1]), dtype=unflat_batch_x.dtype, device=unflat_batch_x.device)
        if (current_dec_inp.shape[0]==unflat_batch_x.shape[0]):
            current_batch_x_mark = self.global_batch_x_mark
            current_batch_y_mark = self.global_batch_y_mark
        else:
            current_dec_inp = current_dec_inp.repeat(unflat_batch_x.shape[0],1,1)
            current_batch_x_mark = self.global_batch_x_mark.repeat(unflat_batch_x.shape[0],1,1)
            current_batch_y_mark = self.global_batch_y_mark.repeat(unflat_batch_x.shape[0],1,1)
            
        current_dec_inp[:,:self.label_len,:] = unflat_batch_x[:,self.label_len:,:]
                
        output = super(Reduced_io_transformer_wrapper, self).forward(unflat_batch_x.float(), current_batch_x_mark.float(), current_dec_inp.float(), current_batch_y_mark.float())
        
        return output[:,self.pred_time_step,:]

    def set_global_batch_x_mark(self, global_batch_x_mark):
        self.global_batch_x_mark = global_batch_x_mark

    def set_global_batch_y_mark(self, global_batch_y_mark):
        self.global_batch_y_mark = global_batch_y_mark

    def set_pred_time_step(self, pred_hor_explained):
        self.pred_time_step = pred_hor_explained

    def get_pred_time_step(self):
        return self.pred_time_step


class Reduced_io_autoformer_wrapper(Autoformer.Model):
    def __init__(self, configs, global_batch_x_mark=0, global_batch_y_mark=0, device="cpu", pred_hor_explained=0):
        self.pred_time_step = pred_hor_explained
        self.global_batch_x_mark = global_batch_x_mark
        self.global_batch_y_mark = global_batch_y_mark
        self.device = device
        self.c_out = configs.c_out
        self.label_len = configs.label_len
        super(Reduced_io_autoformer_wrapper, self).__init__(configs)
        
    def forward(self, input):
        if not torch.is_tensor(input):
            input = torch.from_numpy(input).to(self.device)
        
        unflat_batch_x = input.unflatten(1,[self.pred_len, self.c_out])
        current_dec_inp = torch.zeros((self.global_batch_y_mark.shape[0],self.global_batch_y_mark.shape[1],unflat_batch_x.shape[-1]), dtype=unflat_batch_x.dtype, device=unflat_batch_x.device)
        if (current_dec_inp.shape[0]==unflat_batch_x.shape[0]):
            current_batch_x_mark = self.global_batch_x_mark
            current_batch_y_mark = self.global_batch_y_mark
        else:
            current_dec_inp = current_dec_inp.repeat(unflat_batch_x.shape[0],1,1)
            current_batch_x_mark = self.global_batch_x_mark.repeat(unflat_batch_x.shape[0],1,1)
            current_batch_y_mark = self.global_batch_y_mark.repeat(unflat_batch_x.shape[0],1,1)
            
        current_dec_inp[:,:self.label_len,:] = unflat_batch_x[:,self.label_len:,:]

        # print()
        # print("unflat_batch_x device", unflat_batch_x.get_device())
        # print("current_batch_x_mark device", current_batch_x_mark.get_device())
        # print("current_dec_inp device", current_dec_inp.get_device())
        # print("current_batch_y_mark device", current_batch_y_mark.get_device())
        # print("next(self.parameters()).is_cuda",next(self.parameters()).is_cuda)
                
        output = super(Reduced_io_autoformer_wrapper, self).forward(unflat_batch_x.float(), current_batch_x_mark.float(), current_dec_inp.float(), current_batch_y_mark.float())
        
        return output[:,self.pred_time_step,:]

    def set_global_batch_x_mark(self, global_batch_x_mark):
        self.global_batch_x_mark = global_batch_x_mark

    def set_global_batch_y_mark(self, global_batch_y_mark):
        self.global_batch_y_mark = global_batch_y_mark

    def set_pred_time_step(self, pred_hor_explained):
        self.pred_time_step = pred_hor_explained

    def get_pred_time_step(self):
        return self.pred_time_step


class Reduced_o_DLinear_wrapper(DLinear.Model):
    def __init__(self, configs, pred_hor_explained=0):
        self.pred_time_step = pred_hor_explained
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        super(Reduced_o_DLinear_wrapper, self).__init__(configs)
        
    def forward(self, input):
        if not torch.is_tensor(input):
            input = torch.from_numpy(input).to(exp.device)

        unflat_batch_x = input.unflatten(1,[self.seq_len, self.c_out])

        output = super(Reduced_o_DLinear_wrapper, self).forward(unflat_batch_x)
        
        return output[:,self.pred_time_step,:]

    def set_pred_time_step(self, pred_hor_explained):
        self.pred_time_step = pred_hor_explained

    def get_pred_time_step(self):
        return self.pred_time_step


class Reduced_o_NLinear_wrapper(NLinear.Model):
    def __init__(self, configs, pred_hor_explained=0):
        self.pred_time_step = pred_hor_explained
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        super(Reduced_o_NLinear_wrapper, self).__init__(configs)
        
    def forward(self, input):
        if not torch.is_tensor(input):
            input = torch.from_numpy(input).to(exp.device)

        unflat_batch_x = input.unflatten(1,[self.seq_len, self.c_out])

        output = super(Reduced_o_NLinear_wrapper, self).forward(unflat_batch_x)
        
        return output[:,self.pred_time_step,:]

    def set_pred_time_step(self, pred_hor_explained):
        self.pred_time_step = pred_hor_explained

    def get_pred_time_step(self):
        return self.pred_time_step
