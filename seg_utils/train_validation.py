import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as F
import albumentations as A
from tqdm import tqdm
from seg_utils.utils import label_accuracy_score, add_hist, DenseCRF, batch_crf


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, file_name, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    
    # scheduler
    # scheduler = StepLR(optimizer, 20, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            # outputs = model(images)['out']
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            if isinstance(outputs, list):
                for i in range(len(outputs)):
                    pred = outputs[i]
                    ph, pw = pred.size(2), pred.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        pred = F.interpolate(input=pred, size=(
                            h, w), mode='bilinear', align_corners=True)
                    outputs[i] = pred
            
                loss = 0               
                for i in range(len(outputs)):
                    loss += criterion(outputs[i], masks)
                outputs = outputs[0]

            else:
                loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(data_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir, file_name=file_name)


def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            # outputs = model(images)['out']
            outputs = model(images)
            if isinstance(outputs, list):
                _, outputs = outputs
                ph, pw = outputs.size(2), outputs.size(3)
                h, w = masks.size(1), masks.size(2)
                if ph != h or pw != w:
                    outputs = F.interpolate(input=outputs, size=(
                        h, w), mode='bilinear', align_corners=True)

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , data_loader.dataset.category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss


def save_model(model, saved_dir, file_name='model.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def test(model, test_loader, device, crf_mode=True):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    # DenseCRF 선언
    if crf_mode==True:
        dense_crf = DenseCRF()
        
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            # outs = model(torch.stack(imgs).to(device))['out']
            outs = model(torch.stack(imgs).to(device))

            if crf_mode == True:
                outs = batch_crf(imgs, outs, dense_crf)

            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
            
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


    