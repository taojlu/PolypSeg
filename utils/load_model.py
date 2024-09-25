import torch
import model as model_main
import model_comprise as model_com

device = torch.device('cuda:0')
torch.cuda.set_device(0)


def loading_mode(model_name):
    if model_name == "U_Net":
        model = model_com.U_Net(in_ch=3, out_ch=1).to(device)
    elif model_name == "mamba_pvt":
        model = model_main.pvm_v2_b1().to(device)
    elif model_name == "UltraLight_VM_UNet":
        model = model_main.UltraLight_VM_UNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                               bridge=True).to(device)
    elif model_name == "NestedUNet":
        model = model_com.NestedUNet(in_ch=3, out_ch=1).to(device)
    elif model_name == "VMUNet":
        model = model_com.VMUNet().to(device)
    elif model_name == "UNet_MT":
        model = model_com.UNet_MT(in_dim=3, out_dim=1, num_filters=32).to(device)
    elif model_name == "META_Unet":
        model = model_com.META_Unet().to(device)
    elif model_name == "AttU_Net":
        model = model_com.AttU_Net(img_ch=3, output_ch=1).to(device)
    elif model_name == "M2SNet":
        model = model_com.M2SNet().to(device)
    elif model_name == "MSNet":
        model = model_com.MSNet().to(device)
    elif model_name == "GroupMixFormer":
        model = model_main.GroupMixFormer().to(device)
    elif model_name == "GroupMixFormerSupervise":
        model = model_main.GroupMixFormerSupervise().to(device)
    elif model_name == "SwimGroupMixFormer":
        model = model_main.SwimGroupMixFormer().to(device)
    elif model_name == "SwimGroupFormer":
        model = model_main.SwimGroupFormer().to(device)
    elif model_name == "HighResolutionNetOCR":
        model = model_com.HighResolutionNetOCR().to(device)
    elif model_name == "CFANet":
        model = model_com.CFANet().to(device)
    elif model_name == "PolypPVT":
        model = model_com.polyp_pvt.pvt.PolypPVT().to(device)
    elif model_name == "PVTEncoderNet":
        model = model_main.PVTEncoderNet().to(device)
    elif model_name == "PVTEncoderNet_1":
        model = model_main.PVTEncoderNet_1().to(device)
    elif model_name == "PVT_UNet":
        model = model_main.PVT_UNet().to(device)
    elif model_name == "PVT_UNet_1":
        model = model_main.PVT_UNet_1().to(device)
    elif model_name == "PVT_UNet_NewAtt":
        model = model_main.PVT_UNet_NewAtt().to(device)
    elif model_name == "PVT_UNet_NewAtt_Test":
        model = model_main.PVT_UNet_NewAtt_Test().to(device)
    elif model_name == "PVT_UNet_NewAtt_Test_2":
        model = model_main.PVT_UNet_NewAtt_Test_2().to(device)
    elif model_name == "PVT_UNet_Skip":
        model = model_main.PVT_UNet_Skip().to(device)

    elif model_name == "PvtFormer":
        model = model_main.PvtFormer().to(device)
    elif model_name == "PvtMaFormer":
        model = model_main.PvtMaFormer().to(device)
    elif model_name == "PvtMa1Former":
        model = model_main.PvtMa1Former().to(device)
    elif model_name == "CbamFormer":
        model = model_main.CbamFormer().to(device)
    elif model_name == "MaCbamFormer":
        model = model_main.MaCbamFormer().to(device)
    elif model_name == "MaCbamSuperbisionFormer":
        model = model_main.MaCbamSuperbisionFormer().to(device)
    elif model_name == "MaCbamSuperbisionFormer_1":
        model = model_main.MaCbamSuperbisionFormer_1().to(device)
    elif model_name == "DuattFormer":
        model = model_main.DuattFormer().to(device)
    elif model_name == "PvtSkipCbam":
        model = model_main.PvtSkipCbam().to(device)

    elif model_name == "PvtFormerLight":
        model = model_main.PvtFormerLight().to(device)



    return model
