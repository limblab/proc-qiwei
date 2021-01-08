clc
clear
%file_dir = 'C:\Users\dongq\DeepLabCut\Test-Qiwei-2020-11-23\neural-data\';

%file_name = 'Han_20200922_RT_leftS1_4cameras_joe_qiwei_002';
%file_name = 'CameraDistances_20201123_RW_5_inches_004';
%file_name = 'CameraDistances_20201123_RW_6_inches_006';
%file_name = 'CameraDistances_20201123_RW_7_inches_005';
%file_name = 'CameraDistances_20201123_RW_9_5inches_002';
%file_name = 'CameraDistances_20201123_RW_15_inches_008';



%file_dir = 'C:\Users\dongq\Desktop\CamDistTest_20201207\Neural_data\';
file_dir = 'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\neural-data\';

%file_name = 'CameraDistances_20201207_RW_staggeredDistances003'
file_name = 'Crackle_20201202_RT_LeftS1_4cameras_joe_001'
%file_name = 'CameraDistances_20201207_RW_staggeredDistances003'



file_suffix = '.nev';
%file_suffix = '.ns5';

data_file = strcat(file_dir, file_name, file_suffix);

map_dir = 'C:\Users\dongq\Desktop\20200922\';
map_name = 'SN 6251-001459.cmp';
%map_dir = 'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\neural-data\';
%map_name = 
map_file = strcat(map_dir, map_name);

monkey_name = 'Han';
array_name = 'S1';
task_name = 'RW'; %RT (reaction time) for task I don't do, RW (random walk) for RT3D task?
lab = 6;
ran_by = 'JS';

cds=commonDataStructure();
cds.file2cds(data_file,['array', array_name],...
            ['monkey', monkey_name],lab,'ignoreJumps',['task', task_name], ...
            ['ranBy', ran_by], ['mapFile', map_file]);
%%
cds_kin = cds.kin
cds_analog = cds.analog{1,2}


%%

csv_suffix = '.csv';

file_entire_directory_kin = strcat(file_dir, file_name, '_cds_kin', csv_suffix);
file_entire_directory_analog = strcat(file_dir, file_name, '_cds_analog', csv_suffix);

writetable(cds_kin, file_entire_directory_kin);
writetable(cds_analog, file_entire_directory_analog);

%%
% cds.kin: cds kinematics.
%     t: time
%     still (useless); if the handle is still
%     good (useless); if the encoder data is good
%     x, y: x and y axis of the handle's position
%     vx, vy: x and y speed of the handle
%     ax, ay: x and y acceleration of the handle
%    
% cds.analog: cds analog video syncrhonization data
%     t: time
%     videosync: amplitude of the sync line
%     ainp16 (useless): another analog input
