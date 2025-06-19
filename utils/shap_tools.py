import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

def ChronoSHAP(shap_values_list, input_seq_len, n_feat, feat_names=None, mode=1, pred_hor=0, out_feat=0, custom_cycler=None, grid=0, save=0, path=None, figsize=(12, 5), normalize=True):
    if feat_names==None:
        feat_names = range(n_feat)
        
    if mode==0:
        print("Basic mode")
        fig, axis = plt.subplots(4,2,sharex=True, figsize=figsize) # (7, 9)
        for i in range(len(shap_values_list[pred_hor])):
            signal = shap_values_list[pred_hor][i].reshape((1, input_seq_len, n_feat)).squeeze()
#             print(feat_names[i],signal.shape)
            
            for s in range(signal.shape[1]):
                if i<4:
                    axis[i%4,0].plot(signal[:,s], label = feat_names[s], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][s],linewidth=1.5)
                elif i>=4 and i<8:
                    axis[i%4,1].plot(signal[:,s], label = feat_names[s], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][s],linewidth=1.5)
            if i<4:
                axis[i%4,0].set_title("Output feature: {}".format(feat_names[i]))
#                 axis[i%4,0].legend()
                axis[i%4,0].set_prop_cycle(custom_cycler)
            elif i>=4 and i<=n_feat:
                axis[i%4,1].set_title("Output feature: {}".format(feat_names[i]))
#                 axis[i%4,1].legend()
                axis[i%4,1].set_prop_cycle(custom_cycler)

        # Create a single legend for all subplots
        handles, labels = axis[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=n_feat, fontsize=16)
        fig.suptitle("Time step {}".format(pred_hor), fontsize=16)
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.085)
        if 4*2>n_feat:
            axis[3,1].set_axis_off()
    
    elif mode==1:
        print("ChronoSHAP")
#       prediction horizon, output feature, feature impact (input time step, input feature)
        signals = np.asarray(shap_values_list).squeeze()
        signals = signals.reshape((signals.shape[0], signals.shape[1], input_seq_len, n_feat))
#         print("signals.shape: ",signals.shape)
#         print("\tpred_hor:\t{}\n\tout_feats:\t{}\n\tseq_len:\t{}\n\tinp_feats:\t{}".format(
#             signals.shape[0],signals.shape[1],signals.shape[2],signals.shape[3]))
#         reduction by sum
        accum_signals = np.sum(np.abs(signals), axis=(1,3))
        if normalize:
            for i in range(accum_signals.shape[0]):
                accum_signals[i] = (accum_signals[i]-np.min(accum_signals[i]))/(np.max(accum_signals[i])-np.min(accum_signals[i]))
#         print(feat_names[out_feat],signals[:,out_feat,:].shape)
        
#         print(np.sum(signals, axis=(1,3)).shape)
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(accum_signals, cmap="BuGn")
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel("Accumulated impact", rotation=-90, va="bottom", fontsize=18, labelpad=12)
        cbar.ax.tick_params(labelsize=14)
        
        ax.axis([-0.5, input_seq_len-0.5, -0.5, len(shap_values_list)-0.5])
        
        print("accum_signals.shape",accum_signals.shape)
        ax.set_xticks(np.arange(accum_signals.shape[1],step=5))
        ax.set_yticks(np.arange(accum_signals.shape[0],step=5))
        
#         labels_above_threshold = [label if label%10==0 else '' for label in 
#                                   np.arange(accum_signals.shape[1])]
#         ax.xaxis.set_ticklabels(labels_above_threshold, minor = False)
        
#         labels_above_threshold = [label if label%10==0 else '' for label in 
#                                   np.arange(accum_signals.shape[0])]
#         ax.yaxis.set_ticklabels(labels_above_threshold, minor = False)
        
        ax.tick_params(right= True, top= True, left= True, bottom= True, 
                       labelright= False, labeltop= False, labelleft= True, labelbottom= True, labelsize=14)
#                        labelright= True, labeltop= True, labelleft= True, labelbottom= True)

        ax.set_xlabel("Input sequence time steps", fontsize=18, labelpad=12)
        ax.set_ylabel("Prediction horizon", fontsize=18, labelpad=12)
        plt.tight_layout()

    else:
        print("Mode {} not supported".format(mode))
        
    if grid:
        plt.grid()
        
    if save:
        plt.savefig(path, format="pdf", bbox_inches="tight")

    plt.show()
    return None

# This function provides dictionaries with the best-performing models configurations for each model and each dataset analyzed with SHAP. Note model.args values are hard coded ^_^'
def get_model_args():

    # Transformer models args
    transf_model_args = {}
    exch_model_args = [
        "--is_training", "1", \
        "--seed", "10458", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("exchange_rate"), \
        "--model_id", "{}_96".format("exchange"), \
        "--model", "Transformer", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "8", \
        "--dec_in", "8", \
        "--c_out", "8", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    transf_model_args["exchange_rate"] = exch_model_args

    etth1_model_args = [
        "--is_training", "1", \
        "--seed", "12890", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh1"), \
        "--model_id", "{}_96".format("ETTh1"), \
        "--model", "Transformer", \
        "--data", "{}".format("ETTh1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    transf_model_args["ETTh1"] = etth1_model_args

    etth2_model_args = [
        "--is_training", "1", \
        "--seed", "25565", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh2"), \
        "--model_id", "{}_96".format("ETTh2"), \
        "--model", "Transformer", \
        "--data", "{}".format("ETTh2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    transf_model_args["ETTh2"] = etth2_model_args

    ettm1_model_args = [
        "--is_training", "1", \
        "--seed", "32598", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm1"), \
        "--model_id", "{}_96".format("ETTm1"), \
        "--model", "Transformer", \
        "--data", "{}".format("ETTm1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    transf_model_args["ETTm1"] = ettm1_model_args

    ettm2_model_args = [
        "--is_training", "1", \
        "--seed", "15349", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm2"), \
        "--model_id", "{}_96".format("ETTm2"), \
        "--model", "Transformer", \
        "--data", "{}".format("ETTm2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    transf_model_args["ETTm2"] = ettm2_model_args

    weather_model_args = [
        "--is_training", "1", \
        "--seed", "15227", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("weather"), \
        "--model_id", "{}_96".format("weather"), \
        "--model", "Transformer", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "21", \
        "--dec_in", "21", \
        "--c_out", "21", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    transf_model_args["weather"] = weather_model_args

    ili_model_args = [
        "--is_training", "1", \
        "--seed", "3144", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("national_illness"), \
        "--model_id", "{}_36".format("ili"), \
        "--model", "Transformer", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "36", \
        "--label_len", "18", \
        "--pred_len", "36", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    transf_model_args["national_illness"] = ili_model_args


    # Autoformer models args
    autof_model_args = {}

    exch_model_args = [
        "--is_training", "1", \
        "--seed", "15726", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("exchange_rate"), \
        "--model_id", "{}_96".format("exchange"), \
        "--model", "Autoformer", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "8", \
        "--dec_in", "8", \
        "--c_out", "8", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    autof_model_args["exchange_rate"] = exch_model_args

    etth1_model_args = [
        "--is_training", "1", \
        "--seed", "10458", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh1"), \
        "--model_id", "{}_96".format("ETTh1"), \
        "--model", "Autoformer", \
        "--data", "{}".format("ETTh1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    autof_model_args["ETTh1"] = etth1_model_args

    etth2_model_args = [
        "--is_training", "1", \
        "--seed", "25565", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh2"), \
        "--model_id", "{}_96".format("ETTh2"), \
        "--model", "Autoformer", \
        "--data", "{}".format("ETTh2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    autof_model_args["ETTh2"] = etth2_model_args

    ettm1_model_args = [
        "--is_training", "1", \
        "--seed", "10458", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm1"), \
        "--model_id", "{}_96".format("ETTm1"), \
        "--model", "Autoformer", \
        "--data", "{}".format("ETTm1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    autof_model_args["ETTm1"] = ettm1_model_args

    ettm2_model_args = [
        "--is_training", "1", \
        "--seed", "3293", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm2"), \
        "--model_id", "{}_96".format("ETTm2"), \
        "--model", "Autoformer", \
        "--data", "{}".format("ETTm2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    autof_model_args["ETTm2"] = ettm2_model_args

    weather_model_args = [
        "--is_training", "1", \
        "--seed", "28649", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("weather"), \
        "--model_id", "{}_96".format("weather"), \
        "--model", "Autoformer", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--label_len", "48", \
        "--pred_len", "96", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "21", \
        "--dec_in", "21", \
        "--c_out", "21", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    autof_model_args["weather"] = weather_model_args

    ili_model_args = [
        "--is_training", "1", \
        "--seed", "32598", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("national_illness"), \
        "--model_id", "{}_36".format("ili"), \
        "--model", "Autoformer", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "36", \
        "--label_len", "18", \
        "--pred_len", "36", \
        "--e_layers", "2", \
        "--d_layers", "1", \
        "--factor", "3", \
        "--enc_in", "7", \
        "--dec_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--checkpoints", "F:\Transformers old results\checkpoints"]
    autof_model_args["national_illness"] = ili_model_args


    # DLinear models args
    dl_model_args = {}

    exch_model_args = [
        "--is_training", "1", \
        "--seed", "15349", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("exchange_rate"), \
        "--model_id", "{}_96_96".format("exchange"), \
        "--model", "DLinear", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "8", \
        "--c_out", "8", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    dl_model_args["exchange_rate"] = exch_model_args

    etth1_model_args = [
        "--is_training", "1", \
        "--seed", "15349", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh1"), \
        "--model_id", "{}_96_96".format("ETTh1"), \
        "--model", "DLinear", \
        "--data", "{}".format("ETTh1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    dl_model_args["ETTh1"] = etth1_model_args

    etth2_model_args = [
        "--is_training", "1", \
        "--seed", "3293", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh2"), \
        "--model_id", "{}_96_96".format("ETTh2"), \
        "--model", "DLinear", \
        "--data", "{}".format("ETTh2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    dl_model_args["ETTh2"] = etth2_model_args

    ettm1_model_args = [
        "--is_training", "1", \
        "--seed", "32598", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm1"), \
        "--model_id", "{}_96_96".format("ETTm1"), \
        "--model", "DLinear", \
        "--data", "{}".format("ETTm1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    dl_model_args["ETTm1"] = ettm1_model_args

    ettm2_model_args = [
        "--is_training", "1", \
        "--seed", "10458", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm2"), \
        "--model_id", "{}_96_96".format("ETTm2"), \
        "--model", "DLinear", \
        "--data", "{}".format("ETTm2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    dl_model_args["ETTm2"] = ettm2_model_args

    weather_model_args = [
        "--is_training", "1", \
        "--seed", "10458", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("weather"), \
        "--model_id", "{}_96_96".format("weather"), \
        "--model", "DLinear", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "21", \
        "--c_out", "21", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "16", \
        "--learning_rate", "0.005", \
        "--individual", \
    #     "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    dl_model_args["weather"] = weather_model_args

    ili_model_args = [
        "--is_training", "1", \
        "--seed", "32598", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("national_illness"), \
        "--model_id", "{}_36_36".format("national_illness"), \
        "--model", "DLinear", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "36", \
        "--label_len", "18", \
        "--pred_len", "36", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
    #     "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    dl_model_args["national_illness"] = ili_model_args


    # NLinear models args
    nl_model_args = {}

    exch_model_args = [
        "--is_training", "1", \
        "--seed", "15349", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("exchange_rate"), \
        "--model_id", "{}_96_96".format("exchange"), \
        "--model", "NLinear", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "8", \
        "--c_out", "8", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    nl_model_args["exchange_rate"] = exch_model_args

    etth1_model_args = [
        "--is_training", "1", \
        "--seed", "15726", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh1"), \
        "--model_id", "{}_96_96".format("ETTh1"), \
        "--model", "NLinear", \
        "--data", "{}".format("ETTh1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    nl_model_args["ETTh1"] = etth1_model_args

    etth2_model_args = [
        "--is_training", "1", \
        "--seed", "15349", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTh2"), \
        "--model_id", "{}_96_96".format("ETTh2"), \
        "--model", "NLinear", \
        "--data", "{}".format("ETTh2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    nl_model_args["ETTh2"] = etth2_model_args

    ettm1_model_args = [
        "--is_training", "1", \
        "--seed", "15349", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm1"), \
        "--model_id", "{}_96_96".format("ETTm1"), \
        "--model", "NLinear", \
        "--data", "{}".format("ETTm1"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    nl_model_args["ETTm1"] = ettm1_model_args

    ettm2_model_args = [
        "--is_training", "1", \
        "--seed", "15227", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("ETTm2"), \
        "--model_id", "{}_96_96".format("ETTm2"), \
        "--model", "NLinear", \
        "--data", "{}".format("ETTm2"), \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
        "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    nl_model_args["ETTm2"] = ettm2_model_args

    weather_model_args = [
        "--is_training", "1", \
        "--seed", "15349", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("weather"), \
        "--model_id", "{}_96_96".format("weather"), \
        "--model", "NLinear", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "96", \
        "--pred_len", "96", \
        "--enc_in", "21", \
        "--c_out", "21", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "16", \
        "--learning_rate", "0.005", \
        "--individual", \
    #     "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    nl_model_args["weather"] = weather_model_args

    ili_model_args = [
        "--is_training", "1", \
        "--seed", "28649", \
        "--num_workers", "0", \
        "--root_path", "./dataset/", \
        "--data_path", "{}.csv".format("national_illness"), \
        "--model_id", "{}_36_36".format("national_illness"), \
        "--model", "NLinear", \
        "--data", "custom", \
        "--features", "M", \
        "--seq_len", "36", \
        "--label_len", "18", \
        "--pred_len", "36", \
        "--enc_in", "7", \
        "--c_out", "7", \
        "--des", 'Exp', \
        "--itr", "1", \
        "--train_epochs", "20", \
        "--patience", "5", \
        "--batch_size", "8", \
        "--learning_rate", "0.0005", \
        "--individual", \
    #     "--use_gpu", False, \
        "--checkpoints", "\checkpoints"]
    nl_model_args["national_illness"] = ili_model_args

    return transf_model_args, autof_model_args, dl_model_args, nl_model_args