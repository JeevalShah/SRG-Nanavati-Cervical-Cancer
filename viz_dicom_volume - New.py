import os
import pydicom
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import shutil
from shapely.geometry import Polygon
import scipy.spatial as spatial
from datetime import datetime
from openpyxl import load_workbook

def extract_dicom_metadata(folder_path):
    """
    Extract metadata from DICOM files in the specified folder
    Returns a DataFrame with key metadata and a list of DICOM datasets
    """
    dicom_files = []
    metadata_list = []
    
    # Find all DICOM files in the folder
    for file_path in Path(folder_path).glob('**/*'):
        if file_path.is_file():
            try:
                dicom = pydicom.dcmread(file_path, force=True)
                dicom_files.append(dicom)
                
                # Extract key metadata - removed FilePath, PatientID, StudyDate, ContourCount, ContourNames
                metadata = {
                    'PatientName': str(getattr(dicom, 'PatientName', 'N/A')),
                    'Modality': getattr(dicom, 'Modality', 'N/A'),
                    'SliceLocation': getattr(dicom, 'SliceLocation', 'N/A'),
                    'SliceThickness': getattr(dicom, 'SliceThickness', 'N/A'),
                    'PixelSpacing': getattr(dicom, 'PixelSpacing', ['N/A', 'N/A']),
                    'Rows': getattr(dicom, 'Rows', 'N/A'),
                    'Columns': getattr(dicom, 'Columns', 'N/A'),
                    'BladderVolume_cc': 0,  # Initialize bladder volume
                    'BladderShift_mm': 0   # Initialize bladder shift
                }
                
                # Check for contour data (RT Structure Set)
                if hasattr(dicom, 'Modality') and dicom.Modality == 'RTSTRUCT':
                    # Calculate bladder volume for RT Structure Sets
                    bladder_volume = calculate_bladder_volume(dicom)
                    metadata['BladderVolume_cc'] = bladder_volume
                    metadata['HasContours'] = True
                else:
                    metadata['HasContours'] = False
                
                metadata_list.append(metadata)
                print(f"Processed: {file_path.name}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Create DataFrame from metadata
    df = pd.DataFrame(metadata_list)
    return df, dicom_files

def process_rtstructure(rtss):
    """Extract contour data from an RT Structure Set"""
    contour_data = []
    
    if not hasattr(rtss, 'StructureSetROISequence'):
        return contour_data
    
    # Get ROI names from StructureSetROISequence
    roi_names = {roi.ROINumber: roi.ROIName for roi in rtss.StructureSetROISequence}
    
    # Get contour data from ROIContourSequence
    if hasattr(rtss, 'ROIContourSequence'):
        for roi_contour in rtss.ROIContourSequence:
            roi_number = roi_contour.ReferencedROINumber
            roi_name = roi_names.get(roi_number, f"ROI-{roi_number}")
            
            # Get color if available
            if hasattr(roi_contour, 'ROIDisplayColor'):
                color = roi_contour.ROIDisplayColor
            else:
                color = [255, 0, 0]  # Default to red
            
            # Check for contour data
            contours = []
            if hasattr(roi_contour, 'ContourSequence'):
                contour_count = len(roi_contour.ContourSequence)
                for contour in roi_contour.ContourSequence:
                    if hasattr(contour, 'ContourData'):
                        contours.append(contour.ContourData)
            
            contour_data.append({
                'number': roi_number,
                'name': roi_name,
                'color': color,
                'contour_count': len(contours),
                'contours': contours
            })
    
    return contour_data

def calculate_bladder_volume(rtss_dicom):
    """
    Calculate the volume of bladder from RT structure contour data
    Returns volume in cc (cm^3)
    """
    if not rtss_dicom or rtss_dicom.Modality != 'RTSTRUCT':
        return 0
        
    contour_data = process_rtstructure(rtss_dicom)
    
    # Find bladder structure
    bladder_contours = None
    for structure in contour_data:
        if structure['name'].lower() == 'bladder':
            bladder_contours = structure['contours']
            break
    
    if not bladder_contours:
        return 0
        
    # Calculate volume slice by slice
    volume = 0
    z_map = {}  # To group contours by z-coordinate
    
    # Group contours by Z position (slice)
    for contour in bladder_contours:
        points = np.array(contour).reshape(-1, 3)
        # Use the mean z-value as the key
        z_val = np.mean(points[:, 2])
        if z_val not in z_map:
            z_map[z_val] = []
        z_map[z_val].append(points)
    
    # Get the referenced CT series to determine slice thickness
    slice_thickness = 3.0  # Default in mm
    if hasattr(rtss_dicom, 'ReferencedFrameOfReferenceSequence'):
        for ref_frame in rtss_dicom.ReferencedFrameOfReferenceSequence:
            if hasattr(ref_frame, 'RTReferencedStudySequence'):
                for ref_study in ref_frame.RTReferencedStudySequence:
                    if hasattr(ref_study, 'RTReferencedSeriesSequence'):
                        for ref_series in ref_study.RTReferencedSeriesSequence:
                            if hasattr(ref_series, 'SeriesInstanceUID'):
                                # Try to find slice thickness from referenced series
                                pass  # Would need to lookup the series in available DICOM files
    
    # Calculate area for each slice and multiply by slice thickness
    z_values = sorted(z_map.keys())
    for i in range(len(z_values)):
        z = z_values[i]
        
        # Calculate thickness between slices or use default
        if i < len(z_values) - 1:
            thickness = abs(z_values[i+1] - z)
        else:
            thickness = slice_thickness
        
        # Calculate total area on this slice
        area = 0
        for points in z_map[z]:
            # Extract 2D points (x,y)
            xy_points = points[:, 0:2]
            
            # Create polygon and calculate area
            try:
                polygon = Polygon(xy_points)
                area += polygon.area
            except Exception as e:
                print(f"Error calculating area: {e}")
                
        # Add volume for this slice (area × thickness)
        volume += area * thickness
    
    # Convert to cubic centimeters (cc)
    volume_cc = volume / 1000  # assuming dimensions are in mm
    
    return volume_cc

def calculate_bladder_shift(rtss_dicom_1, rtss_dicom_2):
    """
    Calculate bladder centroid shift between two RT Structure Sets
    Returns shift in mm
    """
    def get_bladder_centroid(rtss_dicom):
        if not rtss_dicom or rtss_dicom.Modality != 'RTSTRUCT':
            return None
            
        contour_data = process_rtstructure(rtss_dicom)
        
        # Find bladder structure
        bladder_contours = None
        for structure in contour_data:
            if structure['name'].lower() == 'bladder':
                bladder_contours = structure['contours']
                break
        
        if not bladder_contours:
            return None
        
        # Calculate centroid from all contour points
        all_points = []
        for contour in bladder_contours:
            points = np.array(contour).reshape(-1, 3)
            all_points.extend(points)
        
        if not all_points:
            return None
            
        all_points = np.array(all_points)
        centroid = np.mean(all_points, axis=0)
        return centroid
    
    centroid_1 = get_bladder_centroid(rtss_dicom_1)
    centroid_2 = get_bladder_centroid(rtss_dicom_2)
    
    if centroid_1 is None or centroid_2 is None:
        return 0
    
    # Calculate Euclidean distance between centroids
    shift = np.linalg.norm(centroid_2 - centroid_1)
    return shift

def update_bladder_shifts(metadata_df, dicom_files):
    """
    Calculate bladder shifts between consecutive RT Structure Sets for the same patient
    and update the DataFrame
    """
    # Get RT Structure Sets
    rtstruct_files = [dicom for dicom in dicom_files if 
                     hasattr(dicom, 'Modality') and dicom.Modality == 'RTSTRUCT']
    
    if len(rtstruct_files) < 2:
        return metadata_df
    
    # Group by patient
    patient_groups = {}
    for i, rtss in enumerate(rtstruct_files):
        patient_name = str(getattr(rtss, 'PatientName', 'N/A'))
        if patient_name not in patient_groups:
            patient_groups[patient_name] = []
        patient_groups[patient_name].append((i, rtss))
    
    # Calculate shifts for each patient
    for patient_name, rt_list in patient_groups.items():
        if len(rt_list) < 2:
            continue
            
        # Sort by study date if available
        rt_list.sort(key=lambda x: getattr(x[1], 'StudyDate', ''))
        
        # Calculate shift between consecutive scans
        for i in range(1, len(rt_list)):
            idx_current, rtss_current = rt_list[i]
            idx_previous, rtss_previous = rt_list[i-1]
            
            shift = calculate_bladder_shift(rtss_previous, rtss_current)
            
            # Find the corresponding row in metadata_df and update shift
            mask = (metadata_df['PatientName'] == patient_name) & \
                   (metadata_df['Modality'] == 'RTSTRUCT')
            
            # Get the indices where this condition is true
            matching_indices = metadata_df[mask].index.tolist()
            
            # Update the shift for the current RT structure (if we can identify it)
            if idx_current < len(matching_indices):
                metadata_df.loc[matching_indices[idx_current], 'BladderShift_mm'] = shift
    
    return metadata_df

def visualize_slice_with_contours(ct_dicoms, rtss_dicom=None, slice_index=0, show_only_bladder=True, save_path=None):
    """
    Visualize a CT slice with overlaid contours if available
    """
    # Sort CT slices by SliceLocation
    ct_dicoms = sorted(ct_dicoms, key=lambda x: 
                      float(getattr(x, 'SliceLocation', 0)) if hasattr(x, 'SliceLocation') else 0)
    
    # Get the specified slice
    if 0 <= slice_index < len(ct_dicoms):
        ct_slice = ct_dicoms[slice_index]
    else:
        print(f"Slice index {slice_index} out of range. Using first slice.")
        ct_slice = ct_dicoms[0]
    
    # Create figure if not in interactive mode
    if plt.get_fignums():
        fig = plt.gcf()
        plt.clf()
    else:
        fig = plt.figure(figsize=(10, 10))
    
    # Display the CT image
    plt.imshow(ct_slice.pixel_array, cmap='gray')
    plt.title(f"CT Slice: {getattr(ct_slice, 'SliceLocation', 'Unknown')}")
    
    # Track if we found bladder contour on this slice
    bladder_found = False
    
    # Overlay contours if RTSTRUCT is available
    if rtss_dicom is not None and rtss_dicom.Modality == 'RTSTRUCT':
        contour_data = process_rtstructure(rtss_dicom)
        
        # Get slice position info
        slice_pos = float(getattr(ct_slice, 'SliceLocation', 0))
        slice_thickness = float(getattr(ct_slice, 'SliceThickness', 1.0))
        pixel_spacing = getattr(ct_slice, 'PixelSpacing', [1, 1])
        rows = int(getattr(ct_slice, 'Rows', 512))
        cols = int(getattr(ct_slice, 'Columns', 512))
        
        # Get the image position patient (origin)
        image_position = getattr(ct_slice, 'ImagePositionPatient', [0, 0, 0])
        
        # Draw contours for this slice
        for structure in contour_data:
            # Skip if we're only showing bladder and this isn't bladder
            if show_only_bladder and structure['name'].lower() != 'bladder':
                continue
                
            for contour in structure['contours']:
                # Convert contour data from flat array to points
                points = np.array(contour).reshape(-1, 3)
                
                # Check if contour is close to this slice
                z_positions = points[:, 2]
                if abs(np.mean(z_positions) - slice_pos) <= slice_thickness/2:
                    # Project 3D points to 2D slice coordinates
                    x_points = (points[:, 0] - image_position[0]) / pixel_spacing[0]
                    y_points = (points[:, 1] - image_position[1]) / pixel_spacing[1]
                    
                    # Draw contour in yellow
                    plt.plot(x_points, y_points, color='yellow', linewidth=2)
                    if structure['name'].lower() == 'bladder':
                        bladder_found = True
    
    plt.axis('off')
    
    # Save the image if requested
    if save_path and bladder_found:
        patient_name = str(getattr(ct_slice, 'PatientName', 'unknown')).replace('^', '_')
        slice_loc = str(getattr(ct_slice, 'SliceLocation', slice_index)).replace('.', '_')
        filename = f"{patient_name}_bladder_slice_{slice_loc}.png"
        save_file = os.path.join(save_path, filename)
        plt.savefig(save_file, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_file}")
    
    # Only display if not saving images in batch mode
    if not save_path:
        plt.draw()

def interactive_slice_viewer(ct_dicoms, rtss_dicom=None):
    """
    Interactive viewer for scrolling through CT slices with contours
    """
    if not ct_dicoms:
        print("No CT slices to display")
        return
    
    # Sort slices by location
    ct_dicoms = sorted(ct_dicoms, key=lambda x: 
                      float(getattr(x, 'SliceLocation', 0)) if hasattr(x, 'SliceLocation') else 0)
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1])
    
    # Set up the axes for the image and slider
    ax_img = plt.subplot(gs[0])
    ax_slider = plt.subplot(gs[1])
    
    # Initialize with the first slice
    slice_index = 0
    ct_slice = ct_dicoms[slice_index]
    img = ax_img.imshow(ct_slice.pixel_array, cmap='gray')
    ax_img.set_title(f"CT Slice: {getattr(ct_slice, 'SliceLocation', 'Unknown')}")
    ax_img.axis('off')
    
    # Create a slider to navigate through slices
    slider = Slider(ax_slider, 'Slice', 0, len(ct_dicoms) - 1, 
                   valinit=0, valstep=1, valfmt='%d')
    
    # Function to update the displayed slice
    def update(val):
        slice_idx = int(slider.val)
        if 0 <= slice_idx < len(ct_dicoms):
            ct_slice = ct_dicoms[slice_idx]
            img.set_array(ct_slice.pixel_array)
            
            # Clear previous contours
            for line in ax_img.lines:
                line.remove()
            
            # Draw new contours if available
            if rtss_dicom is not None and rtss_dicom.Modality == 'RTSTRUCT':
                draw_contours_on_slice(ct_slice, rtss_dicom, ax_img)
            
            ax_img.set_title(f"CT Slice: {getattr(ct_slice, 'SliceLocation', 'Unknown')}")
            fig.canvas.draw_idle()
    
    # Function to draw contours on a specific slice and axis
    def draw_contours_on_slice(ct_slice, rtss_dicom, axis):
        contour_data = process_rtstructure(rtss_dicom)
        
        # Get slice position info
        slice_pos = float(getattr(ct_slice, 'SliceLocation', 0))
        slice_thickness = float(getattr(ct_slice, 'SliceThickness', 1.0))
        pixel_spacing = getattr(ct_slice, 'PixelSpacing', [1, 1])
        image_position = getattr(ct_slice, 'ImagePositionPatient', [0, 0, 0])
        
        # Draw contours for this slice
        for structure in contour_data:
            # Only show Bladder contours
            if structure['name'].lower() != 'bladder':
                continue
                
            for contour in structure['contours']:
                points = np.array(contour).reshape(-1, 3)
                z_positions = points[:, 2]
                
                # Check if contour is close to this slice
                if abs(np.mean(z_positions) - slice_pos) <= slice_thickness/2:
                    # Project 3D points to 2D slice coordinates
                    x_points = (points[:, 0] - image_position[0]) / pixel_spacing[0]
                    y_points = (points[:, 1] - image_position[1]) / pixel_spacing[1]
                    
                    # Draw contour in yellow
                    axis.plot(x_points, y_points, color='yellow', linewidth=2)
    
    # Initial contour drawing
    if rtss_dicom is not None:
        draw_contours_on_slice(ct_slice, rtss_dicom, ax_img)
    
    # Register the update function with the slider
    slider.on_changed(update)
    
    # Add keyboard navigation
    def key_press(event):
        if event.key == 'right' and slider.val < len(ct_dicoms) - 1:
            slider.set_val(slider.val + 1)
        elif event.key == 'left' and slider.val > 0:
            slider.set_val(slider.val - 1)
    
    fig.canvas.mpl_connect('key_press_event', key_press)
    
    plt.tight_layout()
    plt.show()

def save_all_bladder_contours(ct_dicoms, rtss_dicom=None, save_dir="/contours"):
    """
    Save all CT slices with bladder contours to the specified directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Sort CT slices by SliceLocation
    ct_dicoms = sorted(ct_dicoms, key=lambda x: 
                      float(getattr(x, 'SliceLocation', 0)) if hasattr(x, 'SliceLocation') else 0)
    
    print(f"Saving bladder contour images to {save_dir}...")
    
    # Create a temporary figure for saving
    plt.figure(figsize=(10, 10))
    
    saved_count = 0
    for i, ct_slice in enumerate(ct_dicoms):
        # Save each slice with contours
        visualize_slice_with_contours(ct_dicoms, rtss_dicom, slice_index=i, 
                                     show_only_bladder=True, save_path=save_dir)
        saved_count += 1
    
    plt.close()
    print(f"Saved {saved_count} images with bladder contours.")

def map_rt_to_ct(dicom_files):
    """
    Map RT Structure Sets to corresponding CT series
    Returns a dictionary mapping RTStruct files to lists of corresponding CT files
    """
    # Group files by study instance UID and series
    studies = {}
    for dcm in dicom_files:
        if not hasattr(dcm, 'StudyInstanceUID'):
            continue
            
        study_uid = dcm.StudyInstanceUID
        if study_uid not in studies:
            studies[study_uid] = {'RT': [], 'CT': []}
            
        if dcm.Modality == 'RTSTRUCT':
            studies[study_uid]['RT'].append(dcm)
        elif dcm.Modality == 'CT':
            studies[study_uid]['CT'].append(dcm)
    
    # Create mapping of RT to CT files using index in the original list
    # as the key instead of the FileDataset object
    rt_to_ct_map = {}
    
    # For each study, map RT files to CT files
    for study_uid, study_data in studies.items():
        rt_files = study_data['RT']
        ct_files = study_data['CT']
        
        for rt in rt_files:
            # Create a unique key for this RT structure
            rt_key = None
            if hasattr(rt, 'SOPInstanceUID'):
                rt_key = rt.SOPInstanceUID
            else:
                # Find the index of this RT file in the original list
                for i, dcm in enumerate(dicom_files):
                    if dcm is rt:
                        rt_key = f"rt_index_{i}"
                        break
            
            if rt_key is None:
                continue
                
            # Try to find referenced series
            referenced_series = set()
            
            # Look for referenced series UIDs
            if hasattr(rt, 'ReferencedFrameOfReferenceSequence'):
                for ref_frame in rt.ReferencedFrameOfReferenceSequence:
                    if hasattr(ref_frame, 'RTReferencedStudySequence'):
                        for ref_study in ref_frame.RTReferencedStudySequence:
                            if hasattr(ref_study, 'RTReferencedSeriesSequence'):
                                for ref_series in ref_study.RTReferencedSeriesSequence:
                                    if hasattr(ref_series, 'SeriesInstanceUID'):
                                        referenced_series.add(ref_series.SeriesInstanceUID)
            
            # Filter CT files to those in the referenced series
            matching_ct = []
            if referenced_series:
                matching_ct = [ct for ct in ct_files if hasattr(ct, 'SeriesInstanceUID') and ct.SeriesInstanceUID in referenced_series]
            
            # If no matches found, assume all CT files in the study
            if not matching_ct:
                matching_ct = ct_files
            
            # Store RT file and matching CT files in the map
            rt_to_ct_map[rt_key] = {
                'rt_file': rt,
                'ct_files': matching_ct
            }
    
    return rt_to_ct_map

def analyze_rt_structures(dicom_files):
    """
    Analyze all RT Structure Sets, calculate volumes, and map to CT series
    """
    # Find all RT Structure Sets
    rtstruct_files = [dicom for dicom in dicom_files if hasattr(dicom, 'Modality') and dicom.Modality == 'RTSTRUCT']
    
    if not rtstruct_files:
        print("No RT Structure Sets found.")
        return []
        
    # Map RT files to corresponding CT files
    rt_to_ct_map = map_rt_to_ct(dicom_files)
    
    print("\n" + "="*50)
    print("ANALYSIS OF RT STRUCTURE SETS")
    print("="*50)
    
    # List to store RT structure info for later use
    rt_structures_info = []
    
    # Process each RT Structure Set
    for i, rtss in enumerate(rtstruct_files):
        print(f"\n--- RT Structure Set #{i+1} ---")
        
        # Get basic info
        patient_name = str(getattr(rtss, 'PatientName', 'Unknown'))
        study_desc = getattr(rtss, 'StudyDescription', 'Unknown')
        
        print(f"Patient Name: {patient_name}")
        print(f"Description: {study_desc}")
        
        # Calculate bladder volume
        bladder_volume = calculate_bladder_volume(rtss)
        print(f"Bladder Volume: {bladder_volume:.2f} cc")
        
        # Find the key for this RT structure in the map
        rt_key = None
        if hasattr(rtss, 'SOPInstanceUID'):
            rt_key = rtss.SOPInstanceUID
        else:
            # Find using identity
            for key, value in rt_to_ct_map.items():
                if value['rt_file'] is rtss:
                    rt_key = key
                    break
        
        # Find corresponding CT series
        matching_ct = []
        if rt_key and rt_key in rt_to_ct_map:
            matching_ct = rt_to_ct_map[rt_key]['ct_files']
            
        print(f"Corresponding CT Slices: {len(matching_ct)}")
        
        # Print contour information
        contour_data = process_rtstructure(rtss)
        print("Contoured structures:")
        for structure in contour_data:
            print(f"  - {structure['name']} ({structure['contour_count']} contours)")
        
        # Store information for later use
        rt_structures_info.append({
            'index': i, 
            'rtss': rtss,
            'rt_key': rt_key,
            'patient_name': patient_name,
            'bladder_volume': bladder_volume,
            'matching_ct': matching_ct
        })
    
    print("\n" + "="*50)
    return rt_structures_info
def process_single_patient(folder_path):
    """Process a single patient and return the metadata DataFrame"""
    # Extract patient identifier from folder path
    patient_id = os.path.basename(folder_path)
    
    print(f"\nProcessing {patient_id}...")
    
    try:
        # Extract metadata
        metadata_df, dicom_files = extract_dicom_metadata(folder_path)
        
        # Update bladder shifts in the metadata
        metadata_df = update_bladder_shifts(metadata_df, dicom_files)
        
        # Add patient ID as the first column
        metadata_df.insert(0, 'Patient_ID', patient_id)
        
        # Print metadata summary
        print(f"DICOM Metadata Summary for Patient {patient_id}:")
        print(f"Total DICOM files found: {len(metadata_df)}")
        print("Modalities found:")
        print(metadata_df['Modality'].value_counts())
        
        # Print bladder volume summary
        bladder_volumes = metadata_df[metadata_df['BladderVolume_cc'] > 0]['BladderVolume_cc']
        if not bladder_volumes.empty:
            print(f"Bladder Volumes Found:")
            print(f"  Count: {len(bladder_volumes)}")
            print(f"  Mean: {bladder_volumes.mean():.2f} cc")
            print(f"  Range: {bladder_volumes.min():.2f} - {bladder_volumes.max():.2f} cc")
        
        # Print bladder shift summary
        bladder_shifts = metadata_df[metadata_df['BladderShift_mm'] > 0]['BladderShift_mm']
        if not bladder_shifts.empty:
            print(f"Bladder Shifts Found:")
            print(f"  Count: {len(bladder_shifts)}")
            print(f"  Mean: {bladder_shifts.mean():.2f} mm")
            print(f"  Range: {bladder_shifts.min():.2f} - {bladder_shifts.max():.2f} mm")
        
        # Analyze RT structures
        rt_structures_info = analyze_rt_structures(dicom_files)
        ct_files = [dicom for dicom in dicom_files if 
                   hasattr(dicom, 'Modality') and dicom.Modality == 'CT']
        
        if ct_files and rt_structures_info:
            print(f"Found {len(ct_files)} CT slices")
            for i, rt_info in enumerate(rt_structures_info):
                print(f"{i+1}: Patient {rt_info['patient_name']}, Bladder Volume: {rt_info['bladder_volume']:.2f} cc")
        
        return metadata_df
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return None

def main():
    """Main function to process all patients and save to a single Excel file"""
    
    # Define all patient folder paths
    base_path = "C:/Users/G-One/Downloads/Pelvic"
    patient_folders =  ["P1", "P2", "P3", "P4", "P5","P6","P7","P8","P9","P10","P11","P12","P13"]  # Add or remove patient folders as needed
    
    # Alternatively, you can specify full paths directly:
    # patient_paths = [
    #     "C:/Nanavati/P1",
    #     "C:/Nanavati/P2", 
    #     "C:/Nanavati/P3"
    # ]
    
    all_patient_data = []
    processed_patients = []
    
    # Process each patient
    for patient_folder in patient_folders:
        folder_path = os.path.join(base_path, patient_folder)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping...")
            continue
        
        # Process the patient
        patient_data = process_single_patient(folder_path)
        
        if patient_data is not None:
            all_patient_data.append(patient_data)
            processed_patients.append(patient_folder)
    
    # Combine all patient data
    if all_patient_data:
        print(f"\n{'='*50}")
        print("COMBINING ALL PATIENT DATA")
        print(f"{'='*50}")
        
        # Concatenate all DataFrames
        combined_df = pd.concat(all_patient_data, ignore_index=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"all_patients_dicom_metadata_{timestamp}.xlsx"
        
        try:
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                
                # Sheet 1: All patients combined
                combined_df.to_excel(writer, sheet_name='All_Patients', index=False)
                
                # Sheet 2: Individual patient sheets
                for i, (patient_data, patient_id) in enumerate(zip(all_patient_data, processed_patients)):
                    sheet_name = f'{patient_id}'
                    patient_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Sheet 3: Summary statistics
                summary_data = []
                for patient_data, patient_id in zip(all_patient_data, processed_patients):
                    bladder_volumes = patient_data[patient_data['BladderVolume_cc'] > 0]['BladderVolume_cc']
                    bladder_shifts = patient_data[patient_data['BladderShift_mm'] > 0]['BladderShift_mm']
                    
                    summary_data.append({
                        'Patient_ID': patient_id,
                        'Total_Files': len(patient_data),
                        'CT_Files': len(patient_data[patient_data['Modality'] == 'CT']),
                        'RT_Files': len(patient_data[patient_data['Modality'] == 'RTSTRUCT']),
                        'Bladder_Volume_Count': len(bladder_volumes),
                        'Mean_Bladder_Volume_cc': round(bladder_volumes.mean(), 2) if not bladder_volumes.empty else 0,
                        'Min_Bladder_Volume_cc': round(bladder_volumes.min(), 2) if not bladder_volumes.empty else 0,
                        'Max_Bladder_Volume_cc': round(bladder_volumes.max(), 2) if not bladder_volumes.empty else 0,
                        'Bladder_Shift_Count': len(bladder_shifts),
                        'Mean_Bladder_Shift_mm': round(bladder_shifts.mean(), 2) if not bladder_shifts.empty else 0,
                        'Min_Bladder_Shift_mm': round(bladder_shifts.min(), 2) if not bladder_shifts.empty else 0,
                        'Max_Bladder_Shift_mm': round(bladder_shifts.max(), 2) if not bladder_shifts.empty else 0
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Auto-adjust column widths for all sheets
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"\n{'='*50}")
            print("FINAL SUMMARY")
            print(f"{'='*50}")
            print(f"Total patients processed: {len(processed_patients)}")
            print(f"Patients: {', '.join(processed_patients)}")
            print(f"Total DICOM files: {len(combined_df)}")
            print(f"Combined data saved to: {excel_path}")
            print("\nExcel file contains:")
            print("  - 'All_Patients' sheet: Combined data from all patients")
            print("  - Individual patient sheets: Separate data for each patient")
            print("  - 'Summary' sheet: Statistical summary for each patient")
            
        except Exception as e:
            # Fallback to CSV if Excel fails
            csv_path = f"all_patients_dicom_metadata_{timestamp}.csv"
            combined_df.to_csv(csv_path, index=False)
            print(f"\nCombined data saved to: {csv_path} (Excel save failed: {e})")
    
    else:
        print("No patient data was successfully processed.")

if __name__ == "__main__":
    main()