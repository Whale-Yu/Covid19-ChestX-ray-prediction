import torch
from torchvision.models import resnet18


class Recognizer:
    def __init__(self, module_file=""):
        super(Recognizer, self).__init__()
        self.module_file = module_file
        self.CUDA = torch.cuda.is_available()
        self.net = resnet18(pretrained=False, num_classes=3)
        if self.CUDA:
            self.net.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        state = torch.load(self.module_file, map_location=device)
        self.net.load_state_dict(state)
        print("加载模型完毕!")
        self.net.eval()

    @torch.no_grad()
    def recognzie(self, img):
        with torch.no_grad():
            if self.CUDA:
                img = img.cuda()
            # print(pre_img)
            img = img.view(-1, 3, 224, 224)
            y = self.net(img)
            p_y = torch.nn.functional.softmax(y, dim=1)
            p, cls_idx = torch.max(p_y, dim=1)
            return cls_idx.cpu(), p.cpu()
