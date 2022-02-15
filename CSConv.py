
import torch
import torch.nn as nn
import torch.nn.functional as F

class CFG(nn.Module):
    def __init__(self, in_planes, K, T, init_weight=True):
        super(CFG, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = in_planes
        else:
            hidden_planes = K
        self.fc1_1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(hidden_planes)
        self.fc1_2_new = nn.Conv2d(hidden_planes, int(K*in_planes), 1, bias=True)
        self.bn1_2_new = nn.BatchNorm2d(int(K*in_planes))
        self.T = T
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x1 = self.fc1_1(x)
        x1 = self.bn1_1(x1)
        x1 = F.relu(x1)
        x1 = self.fc1_2_new(x1)
        x1 = self.bn1_2_new(x1)
        x1 = x1.view(x.size(0), -1)
        return F.softmax(x1/self.T, 1)

class CSConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3,3), stride=1, padding=1, dilation=1, groups=1, bias=True, K=3,T=34, init_weight=True):
        super(CSConv, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K
        self.bias = bias
        self.Theta = CFG(in_planes, K, T)
        #self.attention
        self.Anchored_weight = nn.Parameter(torch.randn(K, out_planes, in_planes, kernel_size[0], kernel_size[1]), requires_grad=True)
        #self.weight
        if bias:
            self.Anchored_bia = nn.Parameter(torch.zeros(K, out_planes))
        #self.bias
        else:
            self.Anchored_bia = None
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_normal_(self.Anchored_weight[i], mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size, in_planes, height, width = x.size()
        Theta = self.Theta(x)
        x = x.view(1, -1, height, width)
        #regerate cs_weight
        Anchored_weight = self.Anchored_weight.transpose(1,2).contiguous().view((self.K * self.in_planes), -1)
        cs_weight = torch.rand(batch_size,self.K * self.in_planes,self.out_planes * self.kernel_size[0]* self.kernel_size[1]).cuda()
        for i in range(batch_size):
            cs_weight[i] = Theta[i].unsqueeze(1)*Anchored_weight
        cs_weight = cs_weight.view(batch_size, self.K, self.in_planes, self.out_planes, self.kernel_size[0], self.kernel_size[1]).sum(dim=1).transpose(1,2).contiguous().view(-1, self.in_planes, self.kernel_size[0], self.kernel_size[1])
        if self.bias is not None:
            cs_bia = torch.mm(Theta.view(batch_size,self.K,-1).mean(axis=2,keepdim=False), self.Anchored_bia).view(-1)
        #conv with cs_weight
        output = F.conv2d(x, weight=cs_weight, bias=cs_bia, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups*batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
  
class M_CFG(nn.Module):
    def __init__(self, in_planes, K, T, init_weight=True):
        super(M_CFG, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = in_planes
        else:
            hidden_planes = K
        self.fc1_1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(hidden_planes)
        self.fc1_2 = nn.Conv2d(hidden_planes, int(K*in_planes*0.5), 1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(int(K*in_planes*0.5))
        
        self.fc2_1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(hidden_planes)
        self.fc2_2 = nn.Conv2d(hidden_planes, int(K*in_planes*0.5), 1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(int(K*in_planes*0.5))
        self.T = T
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.avgpool(x)
        x1 = self.fc1_1(x)
        x1 = self.bn1_1(x1)
        x1 = F.relu(x1)
        x1 = self.fc1_2(x1)
        x1 = self.bn1_2(x1)
        x1 = x1.view(x.size(0), -1)
        
        x2 = self.fc2_1(x)
        x2 = self.bn2_1(x2)
        x2 = F.relu(x2)
        x2 = self.fc2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = x2.view(x.size(0), -1)
        return F.softmax(x1/self.T, 1), F.softmax(x2/self.T, 1)

class M_CSConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3,3), stride=1, padding=1, dilation=1, groups=1, bias=True, K=3,T=34, init_weight=True):
        super(M_CSConv, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K
        self.bias = bias
        self.Theta = M_CFG(in_planes*2, K, T)
        self.Anchored_weight = nn.Parameter(torch.randn(K, out_planes, in_planes, kernel_size[0], kernel_size[1]), requires_grad=True)
        if bias:
            self.Anchored_bia = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.Anchored_bia = None
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_normal_(self.Anchored_weight[i], mode='fan_out', nonlinearity='relu')
    
    def forward(self, center, boundary):
        concat_feature = torch.cat((center,boundary),dim=1)
        Theta_center, Theta_boundary = self.Theta(concat_feature)
        batch_size, in_planes, height, width = center.size()
        
        center = center.view(1, -1, height, width)
        boundary = boundary.view(1, -1, height, width)
        Anchored_weight = self.Anchored_weight.transpose(1,2).contiguous().view((self.K * self.in_planes), -1)
        cs_weight_center = torch.rand(batch_size,self.K * self.in_planes,self.out_planes * self.kernel_size[0]* self.kernel_size[1]).cuda()
        cs_weight_boundary = torch.rand(batch_size,self.K * self.in_planes,self.out_planes * self.kernel_size[0]* self.kernel_size[1]).cuda()
        for i in range(batch_size):
            cs_weight_center[i] = Theta_center[i].unsqueeze(1)*Anchored_weight
            cs_weight_boundary[i] = Theta_boundary[i].unsqueeze(1)*Anchored_weight
        cs_weight_center = cs_weight_center.view(batch_size, self.K, self.in_planes, self.out_planes, self.kernel_size[0], self.kernel_size[1]).sum(dim=1).transpose(1,2).contiguous().view(-1, self.in_planes, self.kernel_size[0], self.kernel_size[1])
        cs_weight_boundary = cs_weight_boundary.view(batch_size, self.K, self.in_planes, self.out_planes, self.kernel_size[0], self.kernel_size[1]).sum(dim=1).transpose(1,2).contiguous().view(-1, self.in_planes, self.kernel_size[0], self.kernel_size[1])
        if self.bias is not None:
            cs_bia_center = torch.mm(Theta_center.view(batch_size,self.K,-1).mean(axis=2,keepdim=False), self.Anchored_bia).view(-1)
            cs_bia_boundary = torch.mm(Theta_boundary.view(batch_size,self.K,-1).mean(axis=2,keepdim=False), self.Anchored_bia).view(-1)
        
        output_center = F.conv2d(center, weight=cs_weight_center, bias=cs_bia_center, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups*batch_size)
        output_boundary = F.conv2d(boundary, weight=cs_weight_boundary, bias=cs_bia_boundary, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups*batch_size)
        output_center = output_center.view(batch_size, self.out_planes, output_center.size(-2), output_center.size(-1))
        output_boundary = output_boundary.view(batch_size, self.out_planes, output_boundary.size(-2), output_boundary.size(-1))
        return output_center,output_boundary 

class CCM(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3,3), stride=1, padding=1, dilation=1, groups=1, bias=True, K=3,T=34, init_weight=True):
        super(CCM, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K
        self.bias = bias
        self.Theta = M_CFG(in_planes*2, K, T)
        self.Anchored_weight = nn.Parameter(torch.randn(K, out_planes, in_planes, kernel_size[0], kernel_size[1]), requires_grad=True)
        if bias:
            self.Anchored_bia = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.Anchored_bia = None
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_normal_(self.Anchored_weight[i], mode='fan_out', nonlinearity='relu')
    
    def forward(self, center, boundary, mask):
        concat_feature = torch.cat((center,boundary),dim=1)*mask
        Theta_center, Theta_boundary = self.Theta(concat_feature)
        batch_size, in_planes, height, width = center.size()
        
        center = center.view(1, -1, height, width)
        boundary = boundary.view(1, -1, height, width)
        Anchored_weight = self.Anchored_weight.transpose(1,2).contiguous().view((self.K * self.in_planes), -1)
        cs_weight_center = torch.rand(batch_size,self.K * self.in_planes,self.out_planes * self.kernel_size[0]* self.kernel_size[1]).cuda()
        cs_weight_boundary = torch.rand(batch_size,self.K * self.in_planes,self.out_planes * self.kernel_size[0]* self.kernel_size[1]).cuda()
        for i in range(batch_size):
            cs_weight_center[i] = Theta_center[i].unsqueeze(1)*Anchored_weight
            cs_weight_boundary[i] = Theta_boundary[i].unsqueeze(1)*Anchored_weight
        cs_weight_center = cs_weight_center.view(batch_size, self.K, self.in_planes, self.out_planes, self.kernel_size[0], self.kernel_size[1]).sum(dim=1).transpose(1,2).contiguous().view(-1, self.in_planes, self.kernel_size[0], self.kernel_size[1])
        cs_weight_boundary = cs_weight_boundary.view(batch_size, self.K, self.in_planes, self.out_planes, self.kernel_size[0], self.kernel_size[1]).sum(dim=1).transpose(1,2).contiguous().view(-1, self.in_planes, self.kernel_size[0], self.kernel_size[1])
        if self.bias is not None:
            cs_bia_center = torch.mm(Theta_center.view(batch_size,self.K,-1).mean(axis=2,keepdim=False), self.Anchored_bia).view(-1)
            cs_bia_boundary = torch.mm(Theta_boundary.view(batch_size,self.K,-1).mean(axis=2,keepdim=False), self.Anchored_bia).view(-1)
        
        output_center = F.conv2d(center, weight=cs_weight_center, bias=cs_bia_center, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups*batch_size)
        output_boundary = F.conv2d(boundary, weight=cs_weight_boundary, bias=cs_bia_boundary, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups*batch_size)
        output_center = output_center.view(batch_size, self.out_planes, output_center.size(-2), output_center.size(-1))
        output_boundary = output_boundary.view(batch_size, self.out_planes, output_boundary.size(-2), output_boundary.size(-1))
        return output_center,output_boundary