filedir = '/media/truecrypt1/Research/TN_proj/Control/fulldata/';   %  LTN/, RTN/, 
slicenumber = 110:145;
folderNameL = '/media/truecrypt1/Research/TN_proj/dl_proj/label';
folderNameI = '/media/truecrypt1/Research/TN_proj/dl_proj/image';
d = dir([filedir 'C*']);
for i = 1:length(d)
    filename = [filedir '' d(i).name '/FreeSurfer/mri/' d(i).name '_T1_nu.nii.gz'];
    nii = load_nii(filename);
    
    filename = [filedir '' d(i).name '/FreeSurfer/mri/' d(i).name '_T1_aparc_aseg.nii.gz'];
    nii_aseg = load_nii(filename);
    
    ind_L = find(nii_aseg.img==4);
    ind_R = find(nii_aseg.img==43);
    nii_asegnew= zeros(256,256,256);
    nii_asegnew(ind_L) =  1;
    nii_asegnew(ind_R) =  1;
    
    for mm = 1:length(slicenumber)
        Naseg = length(find(nii_asegnew(slicenumber(mm),:,:)>0));
        
        if Naseg > 100
             aseg_matrix = squeeze(nii_asegnew(slicenumber(mm),:,:));
             baseFileName = [d(i).name '_seg_slice' num2str(slicenumber(mm)) '.jpg'];
             fullFileName = fullfile(folderNameL, baseFileName);
             imwrite(aseg_matrix, fullFileName);
             
             img_matrix = squeeze(nii.img(slicenumber(mm),:,:));
             baseFileName = [d(i).name '_slice' num2str(slicenumber(mm)) '.jpg'];
             fullFileName = fullfile(folderNameI, baseFileName);
             imwrite(img_matrix, fullFileName);
        end
    end
end
