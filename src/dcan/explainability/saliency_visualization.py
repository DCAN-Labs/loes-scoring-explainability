import os
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

fairview_ag_dir = "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/"
# Define input files
background_image = os.path.join(fairview_ag_dir, "05-training_ready/subject-00_session-00_space-MNI_brain_mprage_RAVEL.nii.gz")
overlay_image = os.path.join(fairview_ag_dir, "saliency-maps/subject-00_session-00_space-MNI_brain_mprage_RAVEL.nii.gz")
output_image = '/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/saliency_overlays/subject-00_session-00_space-MNI_brain_mprage_RAVEL.png'

# Load images
bg_img = nib.load(background_image)
overlay_img = nib.load(overlay_image)

# Create the overlay plot
display = plotting.plot_roi(
    overlay_img, 
    bg_img=bg_img, 
    colorbar=True, 
    title="Saliency Map Overlay"
)

# Save the figure
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.close()

print(f"Overlayed image saved to: {output_image}")