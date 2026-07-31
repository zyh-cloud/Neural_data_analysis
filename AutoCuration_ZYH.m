
% folder_data = 'C:\Users\ZYH\Desktop\UFE64\spike_20260630\kilosort\20260630_307001';
clc; clear; close;
root_path = 'C:\Users\ZYH\Desktop\UFE64\spike_20260630\kilosort';
setting_filenames = 'C:\Users\ZYH\Desktop\UFE64\settings.json';
% read the settings
userSettings = jsonc.jsoncDecode(fileread(setting_filenames));
dir_list = dir(root_path);

% path_autocuration = 'path_to_AutoCurationKilosort\AutoCurationKilosort';
for i = 1:length(dir_list)
    item = dir_list(i);
    % 排除 . 和 ..，只保留文件夹
    if ~strcmp(item.name, '.') && ~strcmp(item.name, '..') && item.isdir
        folder_data = fullfile(root_path, item.name);
        disp(['正在处理当前文件夹：', item.name]);
        
        %% 1 clean the waveforms in each cluster
        removeNoiseInsideCluster_zyh(folder_data, userSettings);    %修改过

        %% 2 merge similar clusters
        applyPotentialMerges(folder_data, userSettings);

        %% 3 remove clusters which are pure noise, save FR and SNR
        detectNoiseClusters(folder_data, userSettings);             %修改过
        
        % %% 4 remove duplicated clusters
        % removeDuplicatedClusters(folder_data, userSettings);      %不准确

        %% 5 compute and quality metrics
        computeQualityMetrics(folder_data, userSettings);

        %% 6 determine the quality of each cluster
        labelWithQualityMetrics(folder_data, userSettings);

        %% 7 save to cluster_QC.tsv
        load(fullfile(folder_data, "cluster_mean_wf.mat"));
        load(fullfile(folder_data, "FR_SNR_cluster.mat"));
        load(fullfile(folder_data, "QualityMetrics.mat"));
        cluster_group = readtable(fullfile(folder_data, 'cluster_group.tsv'), 'Delimiter', '\t', 'FileType', 'text');
        if any(strcmpi(cluster_group.Properties.VariableNames, 'cluster_id'))
            for k = 1:size(cluster_ids, 1)
                idx = find(cluster_group.cluster_id == cluster_ids(k));
                if isempty(idx)
                    continue
                end

                labels{k} = cluster_group.group{idx};
            end
        end
        ch = nan(numel(cluster_ids), 1);
        idx = find(ismember(cluster_ids, cluster_ID));
        ch(idx) = ch_orig;
        QC = table();
        QC.cluster_ids = cluster_ids;
        QC.ch_orig = ch;
        QC.labels = labels;
        QC.FR = FR;
        QC.SNR = SNR;
        QC.isi_violations = isi_violations;
        QC.amplitude_cutoffs = amplitude_cutoffs;
        QC.presence_ratio = presence_ratio;
        QC.amplitude_median = amplitude_median;
        QC.amplitude_mean_uV = amplitude_mean;
        QC.isolation_distance = isolation_distance;
        QC.d_prime = d_prime;
        QC.nn_miss_rate = nn_miss_rate;
        QC.nn_hit_rate = nn_hit_rate;
        QC.l_ratio = l_ratio;

        writetable(QC, fullfile(folder_data, 'cluster_QC.tsv'), 'Delimiter', '\t', 'FileType', 'text');
        fprintf('cluster_QC.tsv saved successfully\n');
        
        %% 8 realign the spike times
        realignClusterSpikeTimes(folder_data, userSettings);

        %% 9 output to cluster_info.tsv
        updateClusterInfo(folder_data);

    end
end