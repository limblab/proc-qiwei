folder = 'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D-IncreasedLighting\best_two_likelihood\';
file_type = '.mat'
file_name_1 = 'likelihood_best';
file_name_2 = 'likelihood_best_cam';
file_name_3 = 'likelihood_second_best';
file_name_4 = 'likelihood_second_best_cam';

%%

result = to_bento(file_name_1,folder,file_type);
result = to_bento(file_name_2,folder,file_type);
result = to_bento(file_name_3,folder,file_type);
result = to_bento(file_name_4,folder,file_type);

function result = to_bento(file_name,folder, file_type)
    file = load(strcat(folder,file_name,file_type));
    file_fieldnames = fieldnames(file);
    
    file_height = length(file_fieldnames); %10
    file_width = length(getfield(file,file_fieldnames{1})); %num of frames
    file_mat = zeros(file_height, file_width);

    for i = 1:length(file_fieldnames) %10 markers
        file_mat(i,:) = getfield(file,file_fieldnames{i});
    end
    
    %save file to .mat
    result = 0;
    
    save(strcat(folder,file_name,'_mat',file_type),'file_mat');
    result = 1;
end

