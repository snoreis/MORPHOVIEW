# MORPHOVIEW
**M**ultispecies **O**ptimized **R**efractive index matched tissue **P**reparation for **H**igh-resolution **O**ptical imaging, **V**isualization, **I**mage segmentation, and **E**xploration of **W**hole tissues

The purpose of the enclosed Matlab files is to read cell segmented data output from software such as [Cellpose](https://github.com/MouseLand/cellpose) and calculate different cell shape parameters on a cell-by-cell basis.

Since the whole purpose of this pipeline is to analyze large 3D images, a CUDA-capable NVIDIA GPU is **highly** recommended. To utilize the GPU with python, the NVIDIA [driver](https://www.nvidia.com/en-us/drivers/) for the GPU must be installed, and we also recommend installing the CUDA toolkit (for the purposes of this pipeline, we suggest [CUDA 11.8.0](https://developer.nvidia.com/cuda-toolkit-archive)).


### Citation
If you use this protocol, please cite the bioRxiv [paper](https://www.biorxiv.org/content/10.1101/2024.06.26.600880v1):<br />
"Whole Tissue Imaging of Cellular Boundaries at Sub-Micron Resolutions for Automatic Cell Segmentation: Applications in Epithelial Bending of Ectodermal Appendages" <br />
Sam Norris, Jimmy K. Hu, Neil H Shubin<br />
_bioRxiv_ 2024.06.26.600880; doi: https://doi.org/10.1101/2024.06.26.600880


### Image denoising (optional)
Since our images are very large, and often take a very long time to aquire, we are typically unable to reduce the image noise during aquisition. We find one of the best ways to denoise our large confocal stacks is through the use of the very useful [Noise2Void](https://github.com/juglab/n2v) algorithm, which has recently been converted to run using pytorch using the alternative: [CAREamics](https://careamics.github.io/0.1/). To install CAREamics we found that using [conda (miniconda)](https://www.anaconda.com/docs/main) works quite well, and we refer you to the CAREamics github for updated installation instructions. In brief, we run the following in Anaconda:

```
conda create -n careamics python=3.10
conda activate careamics
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install "careamics[examples, wandb, tensorboard, dev]"
pip install jupyter
```
Since there are some packages in the old `Noise2Void` package that we still use, we recommend to also install: `pip install n2v` in the careamics environment.

We have included our Jupyter Notebook, which reads the individual tiles from a large, tiled, Z-stack image aquired on a Zeiss LSM 900 confocal microscope (although this should work with any other microscope) and denoises them. It is **CRITICAL** to input the raw individual tiles, not the stitched image.

### Image Stitching
For large confocal muli-tile confocal images such as ours, we find that the images need to be stitched not only in the XY-plane but the stitching usually requires Z-axis translations as well. The most useful program we've found thus far is the ImageJ/FIJI plugin [BigStitcher](https://imagej.net/plugins/bigstitcher/). But feel free to use whatever stitching software you like! 

### Cellpose Segmentation
[Cellpose](https://github.com/MouseLand/cellpose) is a great way to segment cells from a 3D image volume as long as your cells have a nice cell boundary marker. Again, we refer you to the Cellpose github for updated installation instructions. In brief, we run the following in Anaconda:
```
conda create --name cellpose python=3.10
conda activate cellpose
python -m pip install cellpose[all]
pip uninstall torch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```
In most cases, the Cellpose GUI can be used to analyze the images: `python -m cellpose --Zstack`.

To load the data in MATLAB, save the masks/segments as a PNG/TIF.

Note: We find that for most 3D data, some training data needs to be provided to accurately label cell boundaries. 

### MATLAB Analysis

**INSTALLATION**
Use of the included .m files requires MATLAB to be installed.

**MAIN.m**
The main MATLAB file from which everythin is run is main.m 
We first load the the data in the form of a 3D TIF image file.
```Matlab
 BaseFileName = "n42/convextest/Experiment1_PostProK_n42Shark_01_Section1_Stitched";

 %Raw Image
 RawImage = tiffreadVolume(BaseFileName + "_C2.tif");
 
 %Corresponding segmented cells
 A = tiffreadVolume(BaseFileName + '_masks_04.tif');
```
This code loads both a raw image data and the corresponding masks file. Note: in this example we load only Channel 2, which includes a signal (antibody, phalloidin, DAPI) that we want to quantify in each segment. Loading the `RawImage` is optional.

Next we use MATLAB's built-in [regionprops3](https://www.mathworks.com/help/images/ref/regionprops3.html) to measure a set of properties for each connected component (object) in the 3-D volumetric image. The output masks/segments from `cellpose` can be automatically loaded into `regionprops3` without any modification.  

```Matlab
%% Calculate region properties
B = regionprops3(A, RawImage, "ConvexHull", "Centroid", "Volume", "MeanIntensity", "PrincipalAxisLength", "ConvexVolume", "Solidity", "BoundingBox", "ConvexImage", "SurfaceArea");
```
Here we have included all properties neccesary to run every type of cell shape analysis included. However, if you are only interested in, for example, cell volume, `B = regionprops3(A,"Volume")` would be enough and run much faster. `regionprops3` can be quite slow for very large images.

Although not neccesary, the segmented image can be "smoothed" by converting all the cells into convex hulls, then rewriting the data. We find that the cellpose output can be a bit on the rough side and quite fragmented if the input image data is not great. Some of this can be mitigated by using recently added `flow3D_smooth` (see more information [here](https://cellpose.readthedocs.io/en/latest/do3d.html#segmentation-settings)), but we find the convex hull method to work quite well. The only input to create the convex hull image are the `Solidity`, `BoundingBox`, and `ConvexImage`.

```Matlab
B = regionprops3(A, "Solidity", "BoundingBox", "ConvexImage");
% Calculate Convex region properties
[A_Convex] = CreateConvexHullImage(BaseFileName, A, B, 1);
B_Convex = regionprops3(A_Convex,RawImage, "Centroid", "Volume", "MeanIntensity", "PrincipalAxisLength", "ConvexHull", "SurfaceArea");
```

Next, the information calculated by `regionprops3` is converted into a 3D array with the same dimensions as the input image file. Thus far, we have written the functions to create image arrays of:
- Cell volume (`VolumeCalculation.m`)
- Staining marker concentration per cell (`ActinConcentration.m`)
- The ratio of the major and minor axes of each cell (`MajorMinorRatio.m`)
- The cell length, or longest principle axis length of the cell (`CellLength.m`)
- The 3D cell sphericity (`SphericityCalculation.m`)
- And the cell density (`CellDensityCalculation.m`)

As an example, we can create a 3D array of the cell volume using:
```Matlab
[A_vol] = VolumeCalculation(FileName, A, B, WriteTiff)
```
Where `FileName` is is the base file name with path included (this is used when writing a TIF output), `A` is the segmented image,  `B` is the output from regionprops3 as shown above, and `WriteTiff` is just a bool that instructs the function to write the array to a tiff file or not (1=True, anthying else won't). This can also read the convex hull version as well: `[A_vol_Convex] = VolumeCalculation(FileName, A_Convex, B_Convex, 1);`

For the sphericty calculation, there are instead 4 outputs:
- `B_Sphericity_array` is a list of the indexed cell sphericities simply using the "Volume" and "SurfaceArea" outputs from regionprops3. If the input segments are directly from cellpose and not smoothed using the convex hull approach described above, we find this data to be very noisy and not very useful.
- `B_Sphericity_array_Ellipsoid` is a list of the indexed cell sphericities calculated by approximating the surface area and volume of the cell using the three principle axes calculated by regionprops3. This assumes that every cell is essentially ellipsoidal in shape. While not perfect, it does circumvent the issue of fragmented cell segments output from cellpose. We find the data to look very good and is quite representative.
- `A_Sphericity` is the output 3D image arra of the `B_Sphericity_array` data.
- `A_Sphericity_Ellipsoid` is the output 3D image arra of the `B_Sphericity_array_Ellipsoid` data.

**Outputs:**
For presentations, sometimes it's better to have Gif files instead:
```Matlab
%% Output as Gif
s = sliceViewer(A_vol,"Parent",figure,"Colormap",parula, "DisplayRange",[P_vol(1) P_vol(5)])
colorbar

hAx = getAxesHandle(s);
filenamegif = FileName + "_Volume.gif"
sliceNums = 1:size(A,3);
for idx = sliceNums
    % Update slice number
    s.SliceNumber = idx;
    % Use getframe to capture image
    I = getframe(hAx);
    [indI,cm] = rgb2ind(I.cdata,256);
    % Write frame to the GIF file
    if idx == 1
        imwrite(indI,cm,filenamegif,"gif","Loopcount",inf,"DelayTime", 0.01);
    else
        imwrite(indI,cm,filenamegif,"gif","WriteMode","append","DelayTime", 0.01);
    end
end
```
