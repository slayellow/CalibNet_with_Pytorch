import torch


def train():
    pass


def validate():
    pass


# Get RT Matrix using Exponential Map
def get_RTMatrix_using_exponential_logarithm_mapping(se_vector):

    print("Input Tensor Size : ", se_vector.shape)

    output = []
    for i in range(se_vector.shape[0]):
        vector = se_vector[i][0]
        v = vector[:3]
        w = vector[3:]

        theta = torch.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])

        w_cross = torch.Tensor([0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0])
        w_cross = torch.reshape(w_cross, [3, 3])

        if int(theta) is 0:
            A = 0
            B = 0
            C = 0
        else :
            A = torch.sin(theta) / theta
            B = (1.0 - torch.cos(theta)) / (torch.pow(theta, 2))
            C = (1.0 - A) / (torch.pow(theta, 2))

        w_cross_square = torch.matmul(w_cross, w_cross)

        R = torch.eye(3) + A * w_cross + B * w_cross_square
        Q = torch.eye(3) + B * w_cross + C * w_cross_square

        t = torch.matmul(Q, torch.unsqueeze(v, 1))

        T = torch.cat([R, t], 1)

        output.append(T.tolist())

    print("Output Tensor Size : ", torch.Tensor(output).shape)
    return torch.Tensor(output)


# Get RT Matrix using Euler Angles
def get_RTMatrix_using_EulerAngles(se_vector, order='ZYX'):

    print("Input Tensor Size : ", se_vector.shape)

    output = []
    for i in range(se_vector.shape[0]):
        vector = se_vector[i][0]
        translation = vector[:3]
        rotation = vector[3:]
        rx = rotation[0]; ry = rotation[1]; rz = rotation[2]

        if order is 'ZYX':
            matR = torch.Tensor([ torch.cos(rz)*torch.cos(ry),
                                  torch.cos(rz)*torch.sin(ry)*torch.sin(rx) - torch.sin(rz)*torch.cos(rx),
                                  torch.cos(rz)*torch.sin(ry)*torch.cos(rx) + torch.sin(rz)*torch.sin(rx),
                                  torch.sin(rz)*torch.cos(ry),
                                  torch.sin(rz)*torch.sin(ry)*torch.sin(rx) + torch.cos(rz)*torch.cos(rx),
                                  torch.sin(rz)*torch.sin(ry)*torch.cos(rx) - torch.cos(rz)*torch.sin(rx),
                                  -(torch.sin(ry)),
                                  torch.cos(ry)*torch.sin(rx),
                                  torch.cos(ry)*torch.cos(rx)])
            matR = torch.reshape(matR, [3, 3])

        t = -torch.matmul((torch.transpose(matR, 0, 1)), torch.unsqueeze(translation, 1))

        T = torch.cat([matR, t], 1)

        output.append(T.tolist())

    print("Output Tensor Size : ", torch.Tensor(output).shape)
    return torch.Tensor(output)
