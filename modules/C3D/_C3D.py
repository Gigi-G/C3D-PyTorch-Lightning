import torch
import torch.nn as nn

class C3D(nn.Module):
    """
    The C3D network.
    """
    
    def __init__(self, num_classes:int, pretrained:str=None):
        """Constructor.
        
            Args:
                num_classes (int): Number of classes.
                pretrained (str, optional): Path to pretrained model. Defaults to None.
        """
        super(C3D, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        
        self.fc8 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.relu = nn.ReLU()
        
        self.__init_weight()
        
        if pretrained:
            self.__load_pretrained_weights(pretrained)
            
    def forward(self, x):
        """ Forward method """
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        
        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        
        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)
        
        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        
        h = h.view(-1, 8192)
        
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        
        logits = self.fc8(h)
        
        return logits
    
    def __init_weight(self):
        """Init weight."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)      
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def __load_pretrained_weights(self, pretrained:str) -> None:
        """Load pretrained weights.
        
            Args:
                pretrained (str): Path to pretrained model.
        """
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
                # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias"
        }
        try:
            p_dict = torch.load(pretrained)
            s_dict = self.state_dict()
            for name in p_dict:
                if name not in corresp_name:
                    continue
                s_dict[corresp_name[name]] = p_dict[name]
                
            self.load_state_dict(s_dict)
            
            print("Pretrained model has been loaded.")
        except:
            print("Load pretrained model failed.")
            
    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net except for fc8.
        """
        b = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b, self.conv5a, self.conv5b, self.fc6, self.fc7]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k
    
    def get_10x_lr_params(model):
        """
        This generator returns all the parameters for the last fc layer of the net.
        """
        b = [model.fc8]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k