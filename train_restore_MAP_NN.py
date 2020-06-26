import numpy as np
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
import pickle
import argparse
import yaml
import random
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from restoration import train_run_map_GGNN, train_run_map_NN
from models.shallow_UNET import shallow_UNet
from models.covnet import ConvNet
from datasets import brats_dataset_subj

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=0)
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument('--subjs', type=int, required=True, help="Number of subjects")
    parser.add_argument('--K_actf', type=int, default=1, help="Activation param")

    opt = parser.parse_args()
    name = opt.name
    subj_nbr = opt.subjs
    K_actf = opt.K_actf

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    model_name = config['vae_name']
    data_path = config['path']
    riter = config['riter']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    lr_rate = float(config['lr_rate'])
    step_size = float(config['step_rate'])
    log_freq = config['log_freq']
    original_size = config['orig_size']
    log_dir = config['log_dir']
    n_latent_samples = 25
    epochs = config['epochs']

    validation = True

    print('Name: ', name, 'Lr_rate: ', lr_rate, ' Riter: ', riter, ' Step size: ', step_size)

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load trained vae model
    vae_path = '/scratch_net/biwidl214/jonatank/logs/vae/'
    path = vae_path + model_name + '.pth'
    vae_model = torch.load(path, map_location=torch.device(device))
    vae_model.eval()

    # Create guiding net
    net = shallow_UNet(name, 2, 1, 4).to(device)
    #net = ConvNet(name, 2, 1, 32).to(device)
    #net = UNet(name, 2, 1, 4).to(device)

    #path = '/scratch_net/biwidl214/jonatank/logs/restore/1subj_1e1_1steps_2fch_2MSEloss_pretrain_aug_mask1.pth'
    #net = torch.load(path, map_location=torch.device(device))

    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    # Load list of subjects
    f = open(data_path + 'subj_t2_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list_all = list(subj_dict.keys())
    random.shuffle(subj_list_all)
    subj_list = subj_list_all[:subj_nbr]#['Brats17_CBICA_BFB_1_t2_unbiased.nii.gz'] #
    #if subj_nbr == 1:
    #    subj_list = ['Brats17_2013_14_1_t2_unbiased.nii.gz'] #5 Brats17_TCIA_451_1_t2_unbiased 4 Brats17_CBICA_AUN_1_t2_unbiased 3 Brats17_TCIA_105_1_t2_unbiased 2 Brats17_CBICA_AXW_1_t2_unbiased 1 Brats17_TCIA_241_1_t2_unbiased  0 Brats17_2013_14_1_t2_unbiased

    print(subj_list)

    slices = []
    for subj in subj_list:  # Iterate every subject
        slices.extend(subj_dict[subj])  # Slices for each subject

    if validation:
        random.shuffle(slices)
        train_slices = slices[:int((len(slices)*0.85))]
        valid_slices = slices[int((len(slices)*0.85)):]
        print("Train slices: ", train_slices)
        print("Validation slices: ", valid_slices)
        valid_writer = SummaryWriter(log_dir + "valid_" + name)

    #train_slices = [26224, 25064, 19765, 19730, 25018, 22097, 9462, 15215, 15186, 19762, 9424, 26311, 25100, 19681, 15189, 26270, 19753, 23423, 26236, 15138, 9541, 19534, 25042, 19719, 19693, 22522, 9463, 13036, 22637, 26274, 13060, 19602, 19634, 19759, 13018, 12991, 22011, 26276, 15157, 19577, 22050, 19744, 19557, 25067, 13033, 23339, 19717, 24994, 23390, 19599, 15109, 19626, 22008, 24965, 13066, 19797, 13072, 9474, 22592, 19807, 19553, 13037, 23351, 19685, 19636, 26273, 13024, 13030, 22565, 25033, 23448, 23380, 13067, 13054, 22631, 15212, 22037, 22617, 23458, 22058, 22640, 15155, 22103, 26250, 9511, 19558, 22036, 15210, 26201, 25026, 9422, 15161, 15221, 19746, 9448, 24987, 13025, 24992, 26221, 26323, 15199, 12966, 9420, 22546, 19591, 23433, 15139, 15145, 9536, 22084, 22064, 26307, 19575, 23350, 26325, 13009, 15170, 19616, 9534, 15107, 26248, 23361, 19758, 25079, 22537, 26290, 22105, 12974, 9435, 15194, 25009, 26291, 22095, 19622, 22642, 19805, 22040, 22604, 22023, 26305, 19587, 13081, 23381, 19578, 19563, 25032, 9481, 22570, 22044, 19539, 13088, 13020, 19726, 22020, 25048, 19642, 9498, 15217, 25060, 22622, 13070, 15153, 12986, 19567, 12969, 19706, 19716, 9543, 21997, 9506, 9507, 25022, 15177, 15174, 23329, 22620, 21986, 13028, 22605, 25051, 23441, 19600, 21988, 25041, 19610, 22614, 13027, 19620, 25000, 25004, 26319, 22076, 12982, 25066, 22579, 26220, 15132, 24967, 26227, 25014, 19796, 25089, 22009, 19541, 9491, 13005, 26213, 25070, 19804, 23455, 9535, 19551, 22051, 22535, 19594, 19627, 22074, 15103, 22649, 23362, 19684, 22012, 13079, 22545, 19786, 22045, 22621, 22047, 19799, 26266, 23412, 25072, 19628, 22554, 21976, 22532, 19544, 19569, 9413, 13039, 26313, 15147, 9499, 9454, 19540, 15100, 26251, 22015, 9542, 19586, 22648, 9445, 23363, 9523, 9487, 25015, 15117, 19649, 19583, 25046, 22542, 9475, 25099, 9464, 25029, 22004, 13073, 19763, 22527, 26312, 23392, 19555, 22628, 9545, 25039, 23443, 19803, 23428, 12972, 25078, 13046, 19661, 15113, 15137, 25086, 19808, 26284, 19571, 9415, 23446, 19718, 15198, 19752, 25087, 15152, 19592, 15220, 19658, 25031, 12961, 15227, 13069, 24974, 9471, 21975, 22596, 23331, 15183, 23410, 19542, 12971, 25076, 19712, 19792, 26217, 12954, 19675, 13003, 25058, 22616, 22607, 19604, 9441, 19556, 23373, 13040, 25054, 22584, 26301, 19573, 13012, 19621, 26281, 13007, 24988, 25016, 22091, 26316, 19743, 23399, 15200, 22597, 22634, 24980, 22003, 22080, 13048, 26320, 26218, 19663, 12998, 15134, 22007, 19771, 26333, 23341, 25059, 19532, 19543, 22541, 22633, 13050, 21981, 22536, 9496, 13014, 25021, 15122, 21994, 9472, 23405, 26279, 15164, 23402, 13044, 19734, 19653, 15140, 25081, 19606, 23447, 23349, 9504, 19776, 19652, 15148, 22093, 19619, 25053, 9455, 15203, 15124, 15184, 12983, 19537, 19646, 15209, 19595, 19705, 26231, 13051, 23398, 19659, 26272, 13071, 22010, 15156, 22618, 22043, 13023, 22090, 19536, 22639, 25020, 23404, 13029, 23356, 22033, 26299, 22002, 23343, 21979, 15163, 15143, 22089, 15128, 9509, 25071, 23378, 22019, 22567, 19751, 22056, 15111, 9467, 26269, 22057, 19750, 15125, 22589, 15131, 26286, 22586, 19800, 15223, 19608, 19760, 9538, 13061, 15106, 19697, 19729, 25102, 19549, 15222, 19739, 12964, 9526, 26243, 9452, 19696, 12994, 26208, 22024, 22638, 19692, 15193, 19695, 19580, 15169, 26209, 19564, 23337, 15219, 15127, 13011, 13085, 19603, 19782, 25001, 19699, 15135, 22054, 9520, 26229, 15104, 25073, 9457, 25040, 13075, 23437, 21984, 15185, 15172, 15195, 13022, 25008, 26296, 19794, 15142, 19667, 26238, 26280, 9531, 9418, 25098, 13076, 25037, 13016, 22575, 9438, 19632, 22558, 9497, 9517, 26222, 23401, 24989, 26239, 26332, 23431, 9414, 22538, 15102, 21974, 24996, 19576, 9428, 13078, 15116, 15213, 9478, 23420, 26321, 22075, 15197, 22543, 15129, 22046, 22560, 25043, 26315, 19533, 24997, 26293, 22073, 24966, 22653, 22613, 24972, 25095, 19535, 9492, 19725, 24991, 22085, 26303, 19548, 15173, 19584, 22573, 23345, 22523, 22623, 19785, 19748, 9488, 26302, 23436, 26285, 23413, 15114, 19643, 26215, 15167, 19562, 21980, 19546, 15216, 19767, 24975, 23439, 24986, 15159, 25017, 19656, 26226, 25062, 23415, 22643, 22028, 19579, 9519, 9510, 22088, 22066, 23358, 19687, 23451, 9450, 12978, 21987, 22590, 9425, 25080, 9451, 9484, 22578, 19618, 22061, 15136, 26212, 22576, 25027, 25012, 9446, 26278, 26256, 26202, 22528, 26257, 19742, 22612, 15123, 9530, 19670, 19806, 19598, 25082, 12987, 19612, 9485, 13001, 19727, 26295, 23425, 22563, 25090, 23353, 15112, 22099, 13077, 19722, 25093, 19633, 25085, 22626, 19582, 9524, 23426, 13032, 24995, 19694, 19709, 19775, 12973, 22052, 9532, 22030, 22561, 22588, 19566, 19574, 25030, 23379, 13002, 9540, 22600, 19552, 9453, 19713, 23427, 19601, 12979, 13034, 12996, 19635, 19665, 23395, 25003, 19738, 26252, 9466, 26259, 19630, 9449, 24982, 26306, 22568, 12962, 19559, 22581, 19588, 25077, 22014, 19789, 22106, 9500, 13015, 23442, 19755, 26322, 19617, 23385, 19802, 22063, 25023, 22593, 19690, 13041, 19772, 22553, 23344, 22029, 25056, 15211, 9528, 24969, 22582, 23407, 19768, 19645, 9483, 22654, 26300, 22018, 15149, 22083, 23333, 25019, 19572, 26244, 22566, 21996, 9527, 22632, 13019, 25096, 25097, 9436, 22022, 21989, 26334, 23386, 26310, 15154, 12981, 9423, 19679, 26324, 23366, 12985, 13084, 9513, 24984, 23419, 12975, 12977, 26232, 13010, 13013, 19689, 12955, 19779, 9529, 22572, 21985, 15166, 9503, 22583, 19764, 19529, 22625, 26234, 22574, 23421, 19761, 15179, 19732, 23408, 13021, 26309, 23355, 19711, 15168, 25010, 19728, 23391, 22071, 26277, 23440, 19638, 23367, 26275, 15224, 25049, 22594, 22599, 9544, 15207, 9508, 22556, 12958, 19737, 22067, 26261, 23347, 22104, 15118, 19531, 25075, 26326, 15180, 15108, 22646, 22065, 9444, 19648, 13083, 22591, 26297, 22072, 15218, 21991, 24978, 22630, 15226, 25006, 23371, 23335, 24970, 9477, 19714, 13053, 22048, 15192, 15160, 19597, 15162, 22521, 9456, 9468, 22041, 9440, 25024, 23463, 25038, 9434, 15101, 13045, 19655, 22577, 9465, 15204, 26205, 15165, 22017, 26265, 22540, 24977, 22081, 19641, 19611, 23346, 22636, 23354, 9480, 22585, 9460, 23393, 19770, 19677, 22026, 21992, 12997, 24990, 22025, 19774, 26264, 26240, 19680, 15105, 26216, 22651, 19787, 23460, 25101, 23411, 19660, 19650, 9426, 19673, 26258, 9522, 13052, 12995, 23360, 26214, 26206, 25011, 21999, 22624, 23418, 23403, 22001, 22644, 9502, 19538, 13074, 19651, 22055, 22032, 26329, 25044, 26288, 15187, 25057, 19745, 9430, 23394, 13068, 19590, 22101, 19637, 25005, 22559, 22525, 19629, 24985, 19723, 19623, 9537, 25007, 15178, 19778, 22647, 19757, 26245, 19749, 19715, 26242, 26267, 15175, 25068, 19710, 19731, 22650, 12989, 22000, 19798, 19657, 22096, 9459, 23445, 19530, 19707, 22550, 15151, 19809, 15190, 22086, 25055, 12988, 22013, 23382, 15208, 21990, 19788, 13059, 22611, 15171, 23340, 13047, 23376, 13038, 26254, 23459, 9501, 22038, 22533, 19783, 19741, 19756, 25035, 19708, 13017, 22635, 19720, 9437, 15206, 23452, 23400, 23435, 24993, 23338, 15225, 22595, 19547, 26253, 25069, 13026, 15126, 9458, 19669, 9470, 22555, 19614, 22603, 22569, 19570, 23332, 12990, 23432, 26262, 22078, 22092, 22087, 9417, 19545, 23374, 19704, 23397, 19589, 24998, 19666, 13043, 13031, 19672, 15181, 19691, 19639, 15130, 23384, 26204, 23357, 12953, 15115, 26308, 13064, 23388, 19561, 26318, 9486, 21993, 23417, 19791, 13004, 15201, 9489, 26260, 12992, 15176, 22053, 26304, 22027, 9431, 23457, 15191, 24981, 19647, 22547, 22098, 22094, 21977, 19780, 26327, 19781, 9512, 19735, 22062, 23461, 23450, 12959, 22042, 12999, 26271, 15119, 22598, 26203, 22552, 25002, 22082, 22601, 19777, 22070, 21978, 19615, 26287, 19793, 9419, 15120, 13042, 22530, 9476, 15141, 22039, 23438, 23454, 19631, 15133, 9416, 19625, 24973, 26241, 15188, 23414, 25052, 15202, 26292, 12976, 25013, 26268, 19674, 19784, 23365, 25083, 19754, 15196, 23456, 13006, 23389, 24999, 9432, 26317, 25034, 22035, 26207, 19605, 9439, 13058, 15150, 9494, 22021, 24983, 26328, 15182, 22077, 25074, 23342, 25092, 22529, 26255, 22102, 9443, 26282, 19701, 23330, 25088, 19703, 19550, 13062]

    #valid_slices = [9521, 15214, 22059, 26230, 23387, 9479, 25063, 19668, 19769, 22609, 22641, 9447, 12993, 23359, 22580, 19568, 13082, 26211, 15146, 13087, 22068, 19721, 22534, 19593, 23370, 19795, 26294, 15158, 22571, 22524, 13055, 23429, 13080, 23409, 19683, 19773, 23424, 15099, 9525, 23406, 9473, 15121, 19554, 19664, 25050, 22005, 9539, 26219, 13035, 9518, 19607, 13089, 26283, 15110, 12956, 22034, 25084, 19736, 22627, 12980, 26225, 22544, 23430, 23396, 19700, 23375, 22587, 24979, 24971, 25061, 26228, 19688, 19733, 19654, 26331, 19698, 12960, 13000, 26233, 22107, 19702, 13065, 26314, 22562, 22645, 23453, 26210, 21982, 22615, 12968, 19609, 19613, 19682, 12957, 26289, 19596, 19801, 9421, 9433, 21995, 23352, 12970, 23348, 23369, 12965, 12967, 22531, 19678, 22608, 15144, 19790, 19581, 13049, 22549, 19565, 23462, 9514, 9442, 26298, 9429, 19560, 22079, 22619, 22100, 22602, 25036, 24976, 23336, 26246, 9493, 9516, 24968, 22069, 22539, 23334, 15205, 23434, 25047, 26237, 19662, 13056, 19676, 22557, 25025, 13086, 22629, 23368, 9490, 19747, 9469, 22564, 25065, 22006, 19585, 23364, 25045, 19740, 25091, 19624, 13063, 9533, 22049, 22551, 26223, 9495, 26263, 22606, 25028, 23444, 19671, 19724, 12984, 23383, 26235, 22031, 9482, 21998, 26249, 19686, 23377, 26330, 13057, 23422, 9515, 23449, 13008, 19644, 9505, 22526, 22548, 19640, 12963, 22652, 25094, 26247, 19766, 22060, 23416, 9461, 23372, 9427, 22016, 22610, 21983]


    # Load data
    subj_dataset = brats_dataset_subj(data_path, 'train', img_size, train_slices, use_aug=False)
    subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

    # Load data
    valid_subj_dataset = brats_dataset_subj(data_path, 'train', img_size, valid_slices, use_aug=False)
    valid_subj_loader = data.DataLoader(valid_subj_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    print('Subject ', subj, ' Number of Slices: ', valid_subj_dataset.size)

    # Init logging with Tensorboard
    writer = SummaryWriter(log_dir + name)

    for ep in range(epochs):
        optimizer.zero_grad()
        for batch_idx, (scan, seg, mask) in enumerate(subj_loader):
            # Metrics init
            y_pred = []
            y_true = []

            scan = scan.double().to(device)
            decoded_mu = torch.zeros(scan.size())

            # Get average prior
            for s in range(n_latent_samples):
                with torch.no_grad():
                    recon_batch, z_mean, z_cov, res = vae_model(scan)
                decoded_mu += np.array([1 * recon_batch[i].detach().cpu().numpy() for i in range(scan.size()[0])])

            decoded_mu = decoded_mu / n_latent_samples

            # Remove channel
            scan = scan.squeeze(1)
            seg = seg.squeeze(1)
            mask = mask.squeeze(1)

            restored_batch, loss = train_run_map_GGNN(scan, decoded_mu, net, vae_model, riter, step_size,
                                                      device, writer, seg, mask, K_actf=K_actf, aug=True)

            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('Loss', loss)

            seg = seg.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            # Predicted abnormalty is difference between restored and original batch
            error_batch = np.zeros([scan.size()[0],original_size,original_size])
            restored_batch_resized = np.zeros([scan.size()[0],original_size,original_size])

            for idx in range(scan.size()[0]): # Iterate trough for resize
                error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(), (200,200))
                restored_batch_resized[idx] = resize(restored_batch[idx].cpu().detach().numpy(), (200,200))

            # Remove preds and seg outside mask and flatten
            mask = resize(mask, (scan.size()[0], original_size, original_size))
            seg = resize(seg, (scan.size()[0], original_size, original_size))

            error_batch_m = error_batch[mask > 0].ravel()
            seg_m = seg[mask > 0].ravel().astype(bool)

            # AUC
            y_pred.extend(error_batch_m.tolist())
            y_true.extend(seg_m.tolist())
            if not all(element==0 for element in y_true):
                AUC = roc_auc_score(y_true, y_pred)

            print('AUC : ', AUC)
            writer.add_scalar('AUC:', AUC, ep)
            writer.flush()



        ## VALIDATION
        if validation and ep % 5 == 0:

            for batch_idx, (scan, seg, mask) in enumerate(valid_subj_loader):
                # Metrics init
                valid_y_pred = []
                valid_y_true = []

                scan = scan.double().to(device)
                decoded_mu = torch.zeros(scan.size())

                # Get average prior
                for s in range(n_latent_samples):
                    with torch.no_grad():
                        recon_batch, z_mean, z_cov, res = vae_model(scan)
                    decoded_mu += np.array(
                        [1 * recon_batch[i].detach().cpu().numpy() for i in range(scan.size()[0])])

                decoded_mu = decoded_mu / n_latent_samples

                # Remove channel
                scan = scan.squeeze(1)
                seg = seg.squeeze(1)
                mask = mask.squeeze(1)

                restored_batch, loss = train_run_map_GGNN(scan, decoded_mu, net, vae_model, riter, step_size,
                                                          device, writer, seg, mask, K_actf=K_actf)

                valid_writer.add_scalar('Loss', loss)

                seg = seg.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                # Predicted abnormalty is difference between restored and original batch
                error_batch = np.zeros([scan.size()[0], original_size, original_size])
                restored_batch_resized = np.zeros([scan.size()[0], original_size, original_size])

                for idx in range(scan.size()[0]):  # Iterate trough for resize
                    error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(),
                                              (200, 200))
                    restored_batch_resized[idx] = resize(restored_batch[idx].cpu().detach().numpy(), (200, 200))

                # Remove preds and seg outside mask and flatten
                mask = resize(mask, (scan.size()[0], original_size, original_size))
                seg = resize(seg, (scan.size()[0], original_size, original_size))

                error_batch_m = error_batch[mask > 0].ravel()
                seg_m = seg[mask > 0].ravel().astype(bool)

                # AUC
                valid_y_pred.extend(error_batch_m.tolist())
                valid_y_true.extend(seg_m.tolist())
                if not all(element == 0 for element in valid_y_true):
                    AUC = roc_auc_score(valid_y_true, valid_y_pred)

                print('Valid AUC : ', AUC)
                valid_writer.add_scalar('AUC:', AUC, ep)
                valid_writer.flush()

        if ep % log_freq == 0:
            # Save model
            path = '/scratch_net/biwidl214/jonatank/logs/restore/' + name + str(ep) + '.pth'
            torch.save(net, path)
