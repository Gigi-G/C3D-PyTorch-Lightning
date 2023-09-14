import click
import os
import torch
import numpy as np
from modules.C3D import C3D
import cv2
torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--video', default='./dataset/UCF-101/Bowling/v_Bowling_g02_c03.avi', help='This is the video path.')
@click.option('--output', default='./output', help='This is the output folder.')
@click.option('--device', default='cuda:0', help='This is the device to be used.')
@click.option('-m', default='./models/c3d-pretrained.pth', help='This is the model path.')
@click.option('--classes', default='./output/ucf_labels.txt', help='This is the classes path.')
def main(video:str, output:str, device:str, m:str, classes:str):
    # init device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    # load class names
    with open(classes, 'r') as f:
        class_names = f.readlines()
        f.close()
    
    # init model
    model = C3D(num_classes=101)
    checkpoint = torch.load(m, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # read video and init output
    output = os.path.join(output, video.split('/')[-1])
    print("Input video:", video)
    print("Output video:", output)
    
    cap = cv2.VideoCapture(video)
    retaining = True
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

    clip = []
    print("Reading video...")
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_ITALIC, 0.6,
                        (52, 50, 203), 1)
            cv2.putText(frame, "prob: %.2f" % (probs[0][label] * 100), (20, 40),
                        cv2.FONT_ITALIC, 0.6,
                        (52, 50, 203), 1)
            clip.pop(0)

        out.write(frame)

    cap.release()
    out.release()
    print("Done!")

def center_crop(frame) -> np.ndarray:
    """Crops the given frame at the center.

        Args:
            frame: Frame to be cropped.
            
        Returns:
            Cropped frame.
    """
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

if __name__ == '__main__':
    main()