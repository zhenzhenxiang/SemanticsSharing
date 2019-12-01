# Optical Flow

The optical flow estimation network is based on the implementation of PWC-Net from [RanhaoKang/PWC-Net_pytorch](https://github.com/RanhaoKang/PWC-Net_pytorch).

## Testing

- Prepare image data and model file

  The testing image pairs should be added to the [pred.txt](example/pred.txt) file in example folder. The trained PWC-Net model can be downloaded from [here](https://drive.google.com/file/d/1D_kn5wUljkdLUPY338eiH6LLy-yvfGa9/view?usp=sharing).

- Run the script:
  ```
  ./predict.sh
  ```

The results are in the [example](example) folder, including the warped image, the estimated flow data and the visualized flow image.