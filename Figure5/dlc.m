%% DLC data processing pipeline
%  RD fall 2019

%% Get list of randomly selected videos for dlc training/generating

videoDir = '';
dlcdata.sampleVids (sesData, mainPath);

%% Create dlcdata objects and analysis files for each session

clear

% Read session data from Excel file
mainPath         = 'E:\high_speed_video\';
agPath           = [mainPath, 'aggregate\'];

sesData          = dlcdata.readSesData(mainPath);

for i = 1:height(sesData)
    
    % Initialize dlcdata object
    dd = dlcdata();
    
    % Get session information
    if ~isempty(cell2mat(sesData{i, 1}))
        dd.mouseName           = cell2mat(sesData{i, 1});
        dd.sessionDate         = num2str(sesData{i, 2});
        dd.userData.framerate  = sesData{i, 3};
        dd.userData.preStimDur = sesData{i, 4};
    else
        dd.getSessionInfo(dd);
    end
    
    % Set path
    sessionPath = [mainPath, dd.mouseName, '_', dd.sessionDate];
    
    % Find all .csv files
    csvFiles = dlcdata.findcsvs(sessionPath);
    
    % Load DeepLabCut data
    dd.readcsvs(dd, csvFiles);
    dd.addNans(dd)
    clear csvFiles
    
    % Fix trial mismatch for YT081_191008
    if strcmp(dd.mouseName, 'YT081') && strcmp(dd.sessionDate, '191008')
        dd.RemoveEpochs(1:4)
    end
    
    % Load WaveSurfer data
    wsdata = dd.loadwsdata('mainPath', sessionPath);
    wsdata = dd.getTrialNums(wsdata);
    dd.wspurge(wsdata, dd)
    dd.wsdata2se(wsdata, dd)
    
    % Load BControl data
    bctfile = [sessionPath, '\data_@rdcross3_switchobj_', dd.mouseName,'_', dd.sessionDate, 'a'];
    bctdata = rdcrossmodal_switchArray(bctfile, dd.sessionDate);
    bctdata = dd.purge(bctdata, wsdata);
    dd.loadbctdata(bctdata, dd)
    
    % Clean up workspace
    clear wsdata
    clear bctdata
    clear bctfile
        
    % Pre-processing
    
    % Remove data-points with sudden jumps that are likely mislabelled frames
    dd.removeJumps(dd, 'medfilt')
    
    % Get event times
    dd.getEventTimes(dd)
    
    % Set trial-type map
    dd.ttMap(dd)
    
    % Get whisker angles
    dd.findAngles(dd)
    
    % Save dlcdata object
    disp ('Writing to disk')
    ddfile = fullfile(sessionPath, [dd.mouseName,'_', dd.sessionDate, '_dlcdata.mat']);
    save (ddfile, 'dd', '-v7.3', '-nocompression')
    
    % Processing
    
    % Compute trial-type data
    trialAvgs   = dd.averages (dd);
    trialGroups = dd.groupTrials(dd);
    
    % Save trial-type data
    save([agPath, 'avg\', dd.mouseName, '_', dd.sessionDate, '_avg.mat'], 'trialAvgs');
    save([agPath, 'grp\', dd.mouseName, '_', dd.sessionDate, '_grp.mat'], 'trialGroups');
    
    % Compute and save ROC values
%     trialGroups = load([agPath, 'grp\', dd.mouseName, '_', dd.sessionDate, '_grp.mat']);
%     trialGroups = trialGroups.trialGroups;
    [~, rocData, stdrocData] = dd.getauROCs (trialGroups, 'method', 'builtin', 'framerate', dd.userData.framerate);
    save([agPath, 'roc\', dd.mouseName, '_', dd.sessionDate, '_roc.mat'], 'rocData');
    save([agPath, 'roc\', dd.mouseName, '_', dd.sessionDate, '_stdroc.mat'], 'stdrocData');
    
    % Save ROC figures
    cd ('D:\high_speed_video\figures\')
    dd.plotROCs(dd, rocData)
    
    disp([dd.mouseName, '_', dd.sessionDate, ' done. ', num2str(height(sesData)-i), ' more to go'])
    fprintf('\n');
    
end

%% Post-processing and analysis

% Load dlcdata object
% dd = load();

% Make super-object
DD        = dlcdata.createSuperObj (false);
trialAvgs = dlcdata.averages (DD);

% Plot traces
dd.plotstds()
dd.plotTraces(dd)
dd.plotAvgs (trialAvgs, dd)

%% Secondary analysis for each session

% mainPath = 'E:\data\matlab\dd\';
mainPath = 'E:\data\finkel_et_al_temp\';
sesData  = dlcdata.readSesData(mainPath);

% numVids = 0;
n_trials_tt_rate = 250;
[Hit, Miss, CR, FA] = deal(nan(n_trials_tt_rate, height(sesData)));
% testfolder = 'E:\data\matlab\dd\stdroc\wBonferroni\';

for i = 1:height(sesData)
    if ~isempty(cell2mat(sesData{i, 1}))
        thisMouse   = cell2mat(sesData{i, 1});
        thisSession = num2str(sesData{i, 2});
    else
        continue
    end
    
    sessionName = [thisMouse, '_', thisSession];
    ddfile      = [mainPath, sessionName, '_dlcdata.mat'];
%     rocfile     = [mainPath, sessionName, '_roc.mat'];
%     avgfile     = [mainPath, sessionName, '_avgs.mat'];
    
    try
        load(ddfile);
%         load(rocfile);
%         load(avgfile);
    catch
        error ('Could not load dlcdata object')
    end
    
    
%     dd.baselineAngles (dd)
        
%     trialGroups = dd.groupTrials(dd);
    
%     [~, rocData, stdrocData] = dd.getauROCs (trialGroups, 'method', 'builtin', 'framerate', dd.userData.framerate);
%     [~, ~, stdrocData] = dd.getauROCs (trialGroups, 'method', 'builtin', 'framerate', dd.userData.framerate);
%     
%     onsets = dlcdata.getDiscOnset(rocData, 3);
% 
%     avgData = dd.getAvgs(trialGroups);

    % Trial-type rates across sessions
     ttRates = dd.computeTrialRates(dd);
     Hit(1:n_trials_tt_rate, i) = ttRates.Hit(1:n_trials_tt_rate);
     Miss(1:n_trials_tt_rate, i) = ttRates.Miss(1:n_trials_tt_rate);
     CR(1:n_trials_tt_rate, i) = ttRates.CR(1:n_trials_tt_rate);
     FA(1:n_trials_tt_rate, i) = ttRates.FA(1:n_trials_tt_rate);
     
    % Save dlcdata object
%     disp ('Writing to disk')
%     ddfile = fullfile(mainPath, [dd.mouseName,'_', dd.sessionDate, '_dlcdata.mat']);
%     save (ddfile, 'dd', '-v7.3', '-nocompression')
%     save([mainPath, dd.mouseName, '_', dd.sessionDate, '_roc.mat'], 'rocData');
%     save([mainPath, dd.mouseName, '_', dd.sessionDate, '_stdroc.mat'], 'stdrocData');
%     save([mainPath, thisMouse, '_', thisSession, '_onsets.mat'], 'onsets');
%     save([mainPath, thisMouse, '_', thisSession, '_avgs.mat'], 'avgData');
%     save([mainPath, thisMouse, '_', thisSession, '_rawAvgs_target.mat'], 'avgData');
    
%     save([testfolder, dd.mouseName, '_', dd.sessionDate, '_roc.mat'], 'rocData');
%     save([testfolder, dd.mouseName, '_', dd.sessionDate, '_stdroc.mat'], 'stdrocData');
    
    
    % Save ROC figures
%     cd ('E:\data\figures')
%     dd.plotROCs(dd, rocData)
%     dd.plotrocavgs (dd, rocData, avgData)
%     dlcdata.plotAvgSingle (avgData, thisMouse, thisSession)
    
%     numVids = numVids + dd.numEpochs;
    
    close all
    
end

%% Misc tasks

% Time calibration
[TacLick, TacNoLick, VisLick, VisNoLick] = dd.quickTTMap (dd);

% TacLick   = data(ttData.TacLick, end-1);
% TacNoLick = data(ttData.TacNoLick, end-1);

hold on
t=-57;
time   = linspace(-2443, 3555.2, 3000);
for i = 1:height(TacNoLick)
    dat = cell2mat(TacNoLick{i,1});
    plot(time, dat(1:3000), 'Color', [0 0 0])
%     plot(cell2mat(data{i,1}), cell2mat(data{i,16}))
%     plot([t t], [-15 30], '-k', 'LineStyle', ':', 'Linewidth', 3)
%     plot([t+150 t+150], [-15 30], '-k', 'LineStyle', ':', 'Linewidth', 3)
%     plot([time(1) time(end)]*1000, [10 10], '-k', 'LineStyle', ':', 'Linewidth', 3)
    set(gca, 'xlim', [-200 200])
end


% Sample filenames
folder = 'E:\data\csv\YT084_191015';
dd.getFilenames (TacLick, folder, 5)
dd.getFilenames (TacNoLick, folder, 5)

% TacLick/TacNoLick comparisons
[windDiffs, ~] = dlcdata.getWindScatter('tac', 'tar');
[windDiffs, windAUC] = dlcdata.getWindScatter('tac', 'tar', [14 34]); % Using stimulus window of first 200ms
dlcdata.plotScatter(true, windDiffs)
dlcdata.plotScatter(false, windDiffs, windAUC)
dlcdata.plotSTDscatter (windDiffs, 1) % 1 for target whisker, 0 for surrogate

[windDiffs, windAUC] = dlcdata.getWindScatter('tac', 'sur', [1 200]); % 'tac' or 'vis', 'tar' or 'sur'
dlcdata.plotScatter(false, windDiffs, windAUC)

% Reaction times
[tacRT, visRT] = dlcdata.getReactionTimes ();
dlcdata.plotRT (tacRT, visRT, combined)

%% Figures

dlcdata.plotAvgSingle (avgData)
dlcdata.plotROCSingle (rocData, onsets)
dlcdata.plotScatter (windDiffs)
dlcdata.plotOnsets ()
dlcdata.plotstds ()
dlcdata.plotTTRates (Hit, Miss, CR, FA)
