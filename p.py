# DEFINE PROCESSING SETTINGS
print("---Defining processing settings...")

# Define: Home directory (later this is referred as ".")
HomeDirectory = "~/images/pepper/"

# Define: PhotoScan Project File
PhotoScanProjectFile = ".\\PhotoScan_Project.psz"

# Define: Source images
AerialImagesPattern = "*.jpg"

# Define: AlignPhotosAccuracy ["high", "medium", "low"]
AlignPhotosAccuracy = "medium"

# Define: BuildGeometryQuality ["lowest", "low", "medium", "high", "ultra"]
BuildGeometryQuality = "medium"

# INIT ENVIRONMENT
import os
import glob
import PhotoScan
import math

# Set home folder
os.chdir(HomeDirectory)
print("Home directory: " + HomeDirectory )

# get main app objects
doc = PhotoScan.app.document
app = PhotoScan.Application()

# create chunk
chunk = PhotoScan.Chunk()
chunk.label = "New_Chunk"
doc.chunks.add(chunk)

# FIND ALL PHOTOS IN PATH
AerialImageFiles = glob.glob(AerialImagesPattern)

# LOAD AERIAL IMAGES
print("---Loading images...")
# load each image
for FileName in AerialImageFiles:
   print("File: " + FileName)
   cam = PhotoScan.Camera()
   if not(cam.open(FileName)):
      app.messageBox("Loading of image failed: " + FileName)
   cam.label = cam.path.rsplit("/",1)[1]
   cam.sensor = sensor
   chunk.cameras.add(cam)

# SAVE PROJECT
print("---Saving project...")
print("File: " + PhotoScanProjectFile)
if not(doc.save(PhotoScanProjectFile)):
   app.messageBox("Saving project failed!")

# ALIGN PHOTOS
print("---Aligning photos ...")
print("Accuracy: " + AlignPhotosAccuracy)
chunk.matchPhotos(accuracy=AlignPhotosAccuracy)
chunk.alignPhotos()

# SAVE PROJECT
print("---Saving project...")
print("File: " + PhotoScanProjectFile)
if not(doc.save(PhotoScanProjectFile)):
   app.messageBox("Saving project failed!")

# BUILD GEOMETRY
print("---Building Geometry...")
print("Quality: " + BuildGeometryQuality)
if not(chunk.buildDenseCloud(quality=BuildGeometryQuality)):
   app.messageBox("Builde Dense Cloud failed!")

# SAVE PROJECT
print("---Saving project...")
print("File: " + PhotoScanProjectFile)
if not(doc.save(PhotoScanProjectFile)):
   app.messageBox("Saving project failed!")
  
# SAVE PROJECT
print("---Saving project...")
print("File: " + PhotoScanProjectFile)
if not(doc.save(PhotoScanProjectFile)):
   app.messageBox("Saving project failed!")

# Close photoscan
# app.quit()