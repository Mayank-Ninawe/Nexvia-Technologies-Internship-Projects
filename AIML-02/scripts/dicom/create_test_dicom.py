from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

ds = Dataset()
ds.file_meta = file_meta
ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
ds.PatientName = "TEST^ORTHANC"
ds.PatientID = "ORTHANC-TEST-001"
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()
ds.StudyDate = "20261020"
ds.Modality = "OT"
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.Rows = 256
ds.Columns = 256
ds.BitsAllocated = 16
ds.BitsStored = 16
ds.HighBit = 15
ds.PixelRepresentation = 0
ds.PixelData = (np.arange(256 * 256, dtype=np.uint16) % 4096).tobytes()

output_path = Path(__file__).parents[2] / "backend" / "tests" / "fixtures" / "sample.dcm"
ds.save_as(output_path, write_like_original=False)
print(f"Created {output_path}")
