import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import *
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')

### PEMS08
parser.add_argument('--data',type=str,default='../../data/traf_Datasets/PEMS08_r1_d0_w0_astcgn.npz',help='data path')
parser.add_argument('--adjdata',type=str,default='../../data/traf_Datasets/PEMS08/PEMS08.csv',help='adj data path')
parser.add_argument('--in_dim',type=int,default=5,help='inputs dimension')  ## PEMS08
parser.add_argument('--num_nodes',type=int,default=170,help='number of nodes')  ## PEMS08
parser.add_argument('--save',type=str,default='./garage/PEMS08',help='save path')

### Gothenburg
# parser.add_argument('--data',type=str,default='../../data/traf_Datasets/Gothenburg/All2020_w_astcgn.npz',help='data path')
# parser.add_argument('--adjdata',type=str,default='../../data/traf_Datasets/Gothenburg/tofromcost.csv',help='adj data path')
# parser.add_argument('--in_dim',type=int,default=5,help='inputs dimension')  ## Gothenburg
# parser.add_argument('--num_nodes',type=int,default=61,help='number of nodes')  ## Gothenburg
# parser.add_argument('--save',type=str,default='./garage/Gothenburg',help='save path')

parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=50,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--force', type=str, default=False,help="remove params dir", required=False)
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')

### choose model
# parser.add_argument('--model',type=str,default='gwnet',help='adj type')
# parser.add_argument('--model',type=str,default='H_GCN_wh',help='adj type')
# parser.add_argument('--model',type=str,default='GRCN',help='adj type')
parser.add_argument('--model',type=str,default='ASTGCN_Recent',help='adj type')
# parser.add_argument('--model',type=str,default='ASTGCN_Recent_dynamic',help='adj type')



args = parser.parse_args()
##model repertition
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


    



# def main():
#     #set seed
#     #torch.manual_seed(args.seed)
#     #np.random.seed(args.seed)
#     #load data
#     device = torch.device(args.device)
#     # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
#     adj_mx = util.load_adj(args.adjdata, args.adjtype, num_nodes=args.num_nodes)
#     dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
#     #scaler = dataloader['scaler']
#     supports = [torch.tensor(i).to(device) for i in adj_mx]

#     print(args)
#     if args.model=='gwnet':
#         engine = trainer1( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,\
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='ASTGCN_Recent':
#         engine = trainer2( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='GRCN':
#         engine = trainer3( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='Gated_STGCN':
#         engine = trainer4( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='H_GCN_wh':
#         engine = trainer5( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='OGCRNN':
#         engine = trainer8( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='OTSGGCN':
#         engine = trainer9( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='LSTM':
#         engine = trainer10( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
#     elif args.model=='GRU':
#         engine = trainer11( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                          args.learning_rate, args.weight_decay, device, supports, args.decay
#                          )
   
    
#     # check parameters file
#     params_path=args.save+"/"+args.model
#     if os.path.exists(params_path) and args.force:
#         raise SystemExit("Params folder exists! Select a new params path please!")
#     else:
#         if os.path.exists(params_path):
#             shutil.rmtree(params_path)
#         os.makedirs(params_path)
#         print('Create params directory %s' % (params_path))

#     print("start training...",flush=True)
#     his_loss =[]
#     val_time = []
#     train_time = []
#     for i in range(1,args.epochs+1):
#         #if i % 10 == 0:
#             #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
#             #for g in engine.optimizer.param_groups:
#                 #g['lr'] = lr
#         train_loss = []
#         train_mae = []
#         train_mape = []
#         train_rmse = []
#         t1 = time.time()
#         dataloader['train_loader'].shuffle()
#         for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
#             trainx = torch.Tensor(x).to(device)  # B,N,F,T(Batch, Node, feature, time)
#             trainx= trainx.transpose(1, 2)       # B,F,N,T (Batch, feature, Node, time)
#             # trainx= trainx.transpose(1, 3)      
#             trainy = torch.Tensor(y).to(device)
#             # trainy = trainy.transpose(1, 3)
#             metrics = engine.train(trainx, trainy)
#             train_loss.append(metrics[0])
#             train_mae.append(metrics[1])
#             train_mape.append(metrics[2])
#             train_rmse.append(metrics[3])
#             #if iter % args.print_every == 0 :
#              #   log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
#               #  print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
#         t2 = time.time()
#         train_time.append(t2-t1)
#         #validation
#         valid_loss = []
#         valid_mae = []
#         valid_mape = []
#         valid_rmse = []


#         s1 = time.time()
        
#         for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
#             testx = torch.Tensor(x).to(device)
#             testx= testx.transpose(1, 2)       # B,F,N,T (Batch, feature, Node, time)
#             # testx = testx.transpose(1, 3)
#             testy = torch.Tensor(y).to(device)
#             # testy = testy.transpose(1, 3)
#             # metrics = engine.eval(testx, testy[:,0,:,:])
#             metrics = engine.eval(testx, testy)
#             valid_loss.append(metrics[0])
#             valid_mae.append(metrics[1])
#             valid_mape.append(metrics[2])
#             valid_rmse.append(metrics[3])
#         s2 = time.time()
#         log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
#         print(log.format(i,(s2-s1)))
#         val_time.append(s2-s1)
#         mtrain_loss = np.mean(train_loss)
#         mtrain_mae = np.mean(train_mae)
#         mtrain_mape = np.mean(train_mape)
#         mtrain_rmse = np.mean(train_rmse)

#         mvalid_loss = np.mean(valid_loss)
#         mvalid_mae = np.mean(valid_mae)
#         mvalid_mape = np.mean(valid_mape)
#         mvalid_rmse = np.mean(valid_rmse)
#         his_loss.append(mvalid_loss)

#         log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
#         print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
#         torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
#     print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
#     print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

#     #testing
#     bestid = np.argmin(his_loss)
#     engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
#     engine.model.eval()
    
#     outputs = []
#     realy = torch.Tensor(dataloader['y_test']).to(device)
#     # realy = realy.transpose(1,3)[:,0,:,:]

#     for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
#         testx = torch.Tensor(x).to(device)
#         # testx = testx.transpose(1,3)
#         testx = testx.transpose(1,2)
#         with torch.no_grad():
#             preds,spatial_at,parameter_adj = engine.model(testx)
#             preds=preds.transpose(1,3)
#         outputs.append(preds.squeeze())

#     yhat = torch.cat(outputs,dim=0)
#     yhat = yhat[:realy.size(0),...]


#     print("Training finished")
#     print("The valid loss on best model is", str(round(his_loss[bestid],4)))


#     amae = []
#     amape = []
#     armse = []
#     prediction=yhat
#     for i in range(12):
#         pred = prediction[:,:,i]
#         #pred = scaler.inverse_transform(yhat[:,:,i])
#         #prediction.append(pred)
#         real = realy[:,:,i]
#         metrics = util.metric(pred,real)
#         log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#         print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
#         amae.append(metrics[0])
#         amape.append(metrics[1])
#         armse.append(metrics[2])
    
#     log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#     print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
#     torch.save(engine.model.state_dict(),params_path+"/"+args.model+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
#     prediction_path=params_path+"/"+args.model+"_prediction_results"
#     ground_truth=realy.cpu().detach().numpy()
#     prediction=prediction.cpu().detach().numpy()
#     spatial_at=spatial_at.cpu().detach().numpy()
#     parameter_adj=parameter_adj.cpu().detach().numpy()
#     np.savez_compressed(
#             os.path.normpath(prediction_path),
#             prediction=prediction,
#             spatial_at=spatial_at,
#             parameter_adj=parameter_adj,
#             ground_truth=ground_truth
#         )


if __name__ == "__main__":
    # t1 = time.time()
    # main()
    # t2 = time.time()
    # print("Total time spent: {:.4f}".format(t2-t1))



    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    adj_mx = util.load_adj(args.adjdata, args.adjtype, num_nodes=args.num_nodes)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    #scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)
    if args.model=='gwnet':
        engine = trainer1( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,\
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='ASTGCN_Recent':
        engine = trainer2( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='ASTGCN_Recent_dynamic':
        engine = trainer0( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='GRCN':
        engine = trainer3( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='Gated_STGCN':
        engine = trainer4( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='H_GCN_wh':
        engine = trainer5( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='OGCRNN':
        engine = trainer8( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='OTSGGCN':
        engine = trainer9( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='LSTM':
        engine = trainer10( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='GRU':
        engine = trainer11( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
   
    
    # check parameters file
    params_path=args.save+"/"+args.model
    if os.path.exists(params_path) and args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)  # B,N,F,T(Batch, Node, feature, time)
            trainx= trainx.transpose(1, 2)       # B,F,N,T (Batch, feature, Node, time)
            # trainx= trainx.transpose(1, 3)      
            trainy = torch.Tensor(y).to(device)
            # trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            #if iter % args.print_every == 0 :
             #   log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
              #  print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx= testx.transpose(1, 2)       # B,F,N,T (Batch, feature, Node, time)
            # testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            # testy = testy.transpose(1, 3)
            # metrics = engine.eval(testx, testy[:,0,:,:])
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    engine.model.eval()
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        # testx = testx.transpose(1,3)
        testx = testx.transpose(1,2)
        with torch.no_grad():
            preds,spatial_at,parameter_adj = engine.model(testx)
            preds=preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    prediction=yhat
    for i in range(12):
        pred = prediction[:,:,i]
        #pred = scaler.inverse_transform(yhat[:,:,i])
        #prediction.append(pred)
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(),params_path+"/"+args.model+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    prediction_path=params_path+"/"+args.model+"_prediction_results"
    ground_truth=realy.cpu().detach().numpy()
    prediction=prediction.cpu().detach().numpy()
    spatial_at=spatial_at.cpu().detach().numpy()
    parameter_adj=parameter_adj.cpu().detach().numpy()
    np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            spatial_at=spatial_at,
            parameter_adj=parameter_adj,
            ground_truth=ground_truth
        )
    
    
    
    
    fig, ax = plt.subplots(1,3,figsize=[15,5])
    ax[0].plot(realy.cpu()[:,0,0])
    ax[0].plot(yhat.cpu()[:,0,0])
    ax[0].title.set_text('predict 1 step forward')
    
    ax[1].plot(realy.cpu()[:,0,5])
    ax[1].plot(yhat.cpu()[:,0,5])
    ax[1].title.set_text('predict 6 step forward')
    
    ax[2].plot(realy.cpu()[:,0,11])
    ax[2].plot(yhat.cpu()[:,0,11])
    ax[2].title.set_text('predict 12 step forward')


## PESM08
# GWNET On average over 12 horizons,         Test MAE: 15.7294, Test MAPE: 0.1003, Test RMSE: 24.7152
                                           # Test MAE: 16.3582, Test MAPE: 0.1072, Test RMSE: 25.1460
# H_GCN_wh On average over 12 horizons,      Test MAE: 17.2224, Test MAPE: 0.1173, Test RMSE: 26.3425
                                           # Test MAE: 17.8640, Test MAPE: 0.1162, Test RMSE: 27.4595
# ASTGCN_Recent On average over 12 horizons, Test MAE: 19.3815, Test MAPE: 0.1284, Test RMSE: 29.0444
                                           # Test MAE: 17.1699, Test MAPE: 0.1166, Test RMSE: 26.1573
# GRCN On average over 12 horizons,          Test MAE: 20.4614, Test MAPE: 0.1338, Test RMSE: 32.4759


## Gothenburg(all) 3 days
# GWNET On average over 12 horizons,         Test MAE: 0.0105, Test MAPE: 1.5644, Test RMSE: 0.0185
                                           # Test MAE: 0.0101, Test MAPE: 1.2198, Test RMSE: 0.0183
# H_GCN_wh On average over 12 horizons,      Test MAE: 0.0103, Test MAPE: 1.4281, Test RMSE: 0.0184
                                           # Test MAE: 0.0103, Test MAPE: 1.3377, Test RMSE: 0.0183
# ASTGCN_Recent On average over 12 horizons, Test MAE: 0.0104, Test MAPE: 1.4032, Test RMSE: 0.0181
                                           # Test MAE: 0.0103, Test MAPE: 1.1076, Test RMSE: 0.0185
# GRCN On average over 12 horizons,          Test MAE: 0.0103, Test MAPE: 1.3163, Test RMSE: 0.0187



## Gothenburg(all) 7 days
# GWNET On average over 12 horizons,         Test MAE: 0.0107, Test MAPE: 1.1853, Test RMSE: 0.0193
                                           # Test MAE: 0.0105, Test MAPE: 1.1327, Test RMSE: 0.0191
# H_GCN_wh On average over 12 horizons,      Test MAE: 0.0106, Test MAPE: 1.1122, Test RMSE: 0.0192
                                           # Test MAE: 0.0106, Test MAPE: 1.0636, Test RMSE: 0.0193
# ASTGCN_Recent On average over 12 horizons, Test MAE: 0.0111, Test MAPE: 1.2440, Test RMSE: 0.0201
                                           # Test MAE: 0.0112, Test MAPE: 1.0966, Test RMSE: 0.0210
# GRCN On average over 12 horizons,          Test MAE: 0.0107, Test MAPE: 1.1874, Test RMSE: 0.0196


