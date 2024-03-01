classdef dlcdata < MSessionExplorer
    % For analyzing whisker tracking data from DeepLabCut
    % RD fall 2019
    
    properties
        
        mouseName
        sessionDate
                
    end
    
    
    methods (Access = private)
    end
    
    
    methods (Static)
        
        
        % Pre-processing
        
        
        function sesData = readSesData (mainPath)
            
            if nargin < 1
                mainPath = 'E:\high_speed_video\';
            end
            
            sesData         = cell(2,1);
            [~, ~, sesData] = xlsread([mainPath, 'sesData.xlsx']);
            sesData         = cell2table(sesData(2:end, :));
            
        end
        
        
        function obj = getSessionInfo (obj)
            
            % Open input dialog to get info from user
            prompt          = {'Mouse name:';'Date (yymmdd):';'Capture framerate (Hz, from .avi files):';'Pre-Stim Dur (ms):'};
            dlgtitle        = 'Which session would you like to process?';
            dims            = [1 65];
            definput        = {'RD','19','500.15','2500'};
            answer          = inputdlg(prompt,dlgtitle,dims,definput);
            
            % Set object properties
            obj.mouseName            = answer{1};
            obj.sessionDate          = answer{2};
            obj.userData.framerate   = str2double(answer{3});
            obj.userData.preStimDur  = str2double(answer{4});

        end
        
        
        function csvFiles = findcsvs (directory)
            
            if nargin < 1
                directory = cd;
            end
            
            directory  = [directory, '\csv'];
            d          = dir(directory);
            csvFiles   = {};
            numMatches = 0;
            
            for i=1:length(d)
                % Look for occurences of ['.csv' extension] in the file name
                extIndices = strfind(d(i).name,'.csv');
                
                % If the file is not a directory, and the file has at least one occurence
                if ~d(i).isdir && ~isempty(extIndices)
                    numMatches           = numMatches + 1;
                    csvFiles{numMatches} = fullfile(directory,d(i).name);
                end
            end
            
            disp(['Found ', num2str(numMatches), ' .csv files']);
        
        end
        
        
        function readcsvs (obj, csvFiles)
            % Read all .csv files, organize into a table and load on to
            % dlcdata object
            % RD fall 19
            
            disp ('Loading DeepLabCut data');
            
            % Initialize table
            varNames = {'tarBase_x', 'tarBase_y', 'tarMid_x', 'tarMid_y', 'tarEnd_x', 'tarEnd_y',...
                'surBase_x', 'surBase_y', 'surMid_x', 'surMid_y', 'surEnd_x', 'surEnd_y',...
                'nose_x', 'nose_y'};
            tb = cell(length(csvFiles), length(varNames));
            
            % Read files
            for i=1:length(csvFiles)
                data     = readmatrix(cell2mat(csvFiles(i)));
                tb{i,1}  = data(:,2);
                tb{i,2}  = data(:,3);
                tb{i,3}  = data(:,5);
                tb{i,4}  = data(:,6);
                tb{i,5}  = data(:,8);
                tb{i,6}  = data(:,9);
                tb{i,7}  = data(:,11);
                tb{i,8}  = data(:,12);
                tb{i,9}  = data(:,14);
                tb{i,10} = data(:,15);
                tb{i,11} = data(:,17);
                tb{i,12} = data(:,18);
                tb{i,13} = data(:,20);
                tb{i,14} = data(:,21);
            end
            
            % Set time
            t = cellfun(@(x) cumsum(ones(length(x), 1)) * (1000/obj.userData.framerate),...
                tb(:,1), 'UniformOutput', false);
            t = cellfun(@(x) x - 2500, t, 'UniformOutput', false);
            
            % Update dlcdata
            tb = cell2table([t, tb], 'VariableNames', ['time',varNames]);
            
            % Discard first trial
            tb = tb(2:end, :);
            
            obj.SetTable('DLCData', tb, 'timeSeries');
            
        end
        
        
        function addNans (obj)
            
            if obj.userData.preStimDur < 2500
                missTime   = 2500 - obj.userData.preStimDur;
                missFrames = round(missTime / (1000 / obj.userData.framerate));
                nanArray   = nan(missFrames, 1);
                
                [dlcdata] = obj.GetTable ('DLCData');
                
                for data = 1:width(dlcdata)
                    for trial = 1:height(dlcdata)
                        thisTrial            = dlcdata(trial, data);
                        thisTrial            = table2array(thisTrial{1,1});
                        thisTrial            = [nanArray; thisTrial];
                        dlcdata(trial, data) = {[thisTrial]};
                    end
                end
                
                obj.SetTable('DLCData', dlcdata, 'timeSeries');
            end
        end
        
        
        function resultStruct = loadwsdata (varargin)
            % Load data from ws .h5 file
            
            p = inputParser();
            p.addParameter('mainPath', [], @ischar);
            p.addParameter('delimiterChan', 1, @isscalar);
            p.addParameter('downsampleFactor', 1, @isscalar);
            
            p.parse(varargin{:});
            mainPath      = p.Results.mainPath;
            delimiterChan = p.Results.delimiterChan;
            dsFactor      = p.Results.downsampleFactor;
                  
            % Look for ws .h5 file in main dir
            matches = [];
            if ~isempty(mainPath)
                d       = dir(mainPath);
                matches = arrayfun(@(x) strfind(x.name, '.h5'), d, 'UniformOutput', false);
                wsPath  = [mainPath, '\', (d((cellfun(@isempty, matches)) == 0).name)];
            end
            matches = cell2mat(matches);
            
            % If main dir not specified or more than one .h5 file found
            if isempty(wsPath) || length(matches) > 1
                wsPath = MBrowse.File('D:\high_speed_video', 'Please select a WaveSurfer data file', {'*.h5'});
            end
            
            if isempty(wsPath)
                return;
            else
                disp('Loading WaveSurfer results file');
                wsRawData = ws.loadDataFile(wsPath);
            end
                        
            % Get time series
            wsFiledNames = fieldnames(wsRawData);
            channelNames = wsRawData.header.Acquisition.ActiveChannelNames;
                
            for i = length(wsFiledNames) : -1 : 2
                analogScans{i-1,1} = wsRawData.(wsFiledNames{i}).analogScans;
            end
            
            if ~isempty(delimiterChan)
                % Concatenate sweeps (though usually just one) if using delimiter signal
                analogScans   = {cat(1, analogScans{:})};
                delimiterScan = analogScans{1}(:, delimiterChan) > 4; % threshold analog signal to logical values
            end
            
            % Down-sampling
            rawSampleRate = wsRawData.header.Acquisition.SampleRate;
            
            if dsFactor > 1
                fprintf('Downsampling time series by %d-fold.\n', dsFactor);
                
                sampleRate = rawSampleRate / dsFactor;
                
                % Go through each continuous time series and each channel
                for k = length(analogScans) : -1 : 1
                    for i = size(analogScans{k}, 2) : -1 : 1
                        analogScansDS{k}(:,i) = decimate(analogScans{k}(:,i), dsFactor);
                    end
                end
                
                analogScans = analogScansDS;
                clear analogScansDS;
                
            else
                if dsFactor <= 1
                    fprintf('Downsampling factor is %d; no downsampling is applied\n', dsFactor);
                end
                sampleRate = rawSampleRate;
            end
            
            
            % Parse trials by delimiter times
            if ~isempty(delimiterChan)
                
                % Parse trials from a continuous signal by a delimiter channel
                fprintf('Parsing trials at rising edges in delimiter channel %d\n', delimiterChan);
                
                % Find potential trial onsets
                trialStarts                               = zeros(size(delimiterScan, 1),1);
                oneInds                                   = find(delimiterScan(:,1)==1); %Find indices of ones
                trialStarts(oneInds(1)-(rawSampleRate/2)) = 1; % Parse trials from 500 ms before BitCode
                counter                                   = 0;
                
                for i=2:numel(oneInds)
                    tmp = oneInds(i) - oneInds(i-1);
                    ind = oneInds(i);
                    if tmp > rawSampleRate  % Inter-trial interval is larger than 1 sec
                        trialStarts(ind-(rawSampleRate/5)) = 1;
                        counter = counter+1;
                    elseif i == numel(oneInds)
                        continue
                    end
                end
                
                fprintf('%d trials found\n', counter+1);
                     
                trialStartInds        = find(trialStarts == 1);
                delimiterRiseTimeInMs = trialStartInds / rawSampleRate * 1000;
                
                % Apply delimiters
                timeInMs = (1 : size(analogScans{1},1))' / sampleRate * 1000;
                beginIdx = length(timeInMs) + 1;
                
                for i = length(delimiterRiseTimeInMs) : -1 : 1
                    % Find sample index range
                    endIdx   = beginIdx - 1;
                    beginIdx = find(timeInMs >= delimiterRiseTimeInMs(i), 1, 'first');
                    
                    % Get the number of samples
                    trialLength(i) = endIdx - beginIdx + 1;
                end
                
                analogScans{1}(1:beginIdx-1, :) = [];
                
            else
                % Each sweep is a trial
                disp('The delimiter channel is not specified. Each sweep is treated as a trial.');
                delimiterValue = (1 : length(analogScans))';
                trialLength    = cellfun(@(x) size(x,1), analogScans);
            end
            
            trialLength = trialLength(:);
                  
            % Organize data into table
            analogScans = cell2mat(analogScans);
            analogScans = mat2cell(analogScans, trialLength, ones(1, size(analogScans,2)));
            timeInMs    = arrayfun(@(x) (1:x)' / sampleRate * 1000, trialLength, 'Uni', false);
            tb          = cell2table([timeInMs, analogScans], 'VariableNames', [{'time'}; channelNames]);
            
            % Disregard last trial
            tb = tb(1:end-1, :);
                        
            % Save data info
            resultStruct.wsTable           = tb;
            resultStruct.wsInfo.header     = wsRawData.header;
            resultStruct.wsInfo.sampleRate = sampleRate;
            
            if exist('trialLength', 'var')
                resultStruct.wsInfo.trialLength = trialLength;
            end
            
            if exist('delimiterValue', 'var')
                resultStruct.wsInfo.delimiterValue = delimiterValue(trialMask);
            end
            
        end
        
        
        function wavesurferData = getTrialNums (wavesurferData)
            % Convert BitCode channel on wavesurfer data to trial nums
            % RD summer 2019
            
            digitalBitCode                 = dlcdata.binarize(wavesurferData.wsTable.slid_bitcode, 4); % if bitcode on analog channel
            wsTrialNums                    = dlcdata.readBitCode(digitalBitCode, wavesurferData.wsInfo.sampleRate);
            wavesurferData.wsTable.trialNo = [wsTrialNums];
                        
        end
        
        
        function digSignal = binarize (analogSignal, threshold)
            %
            % To be used in case bitcode was acquired
            % on an analog channel on wavesurfer.
            % Takes analog signal as input and returns
            % binarized signal based on user-defined threshold
            %
            % RD summer 2019
            %
            
            % Get threshold value if not already provided
            if nargin < 2
                answer    = inputdlg({'Threshold:'},'Please enter threshold for binarizing', [1,58], {'4'});
                threshold = str2double(answer{1});
            end
            
            % Initialize empty cell array
            digSignal = cell(length(analogSignal), 1);
                  
            % Populate the array with value 1 for every input signal value
            % over threshold
            for i = 1:length(analogSignal)
                digSignal{i,1} = double(analogSignal{i,1} > threshold);
            end
        end
        
        
        function trialNums = readBitCode (digSignal, sampleRate, bitTime, gapTime)
            
            % Returns trial number signaled by digital BitCode channel in
            % all parsed trials. Does not assume BitCode starts at a defined
            % point
            %
            % RD summer 2019
            
            if ~any(cell2mat(digSignal) == 1)
                disp('This session does not have BitCodes');
                return
            end
            
            if nargin < 3
                bitTime = 42/sampleRate;
                gapTime = 104/sampleRate;
            end
            
            % Set BitCode window parameters and make bit indices
            nbits        = 10+1; % First bit signals trial start, followed by 10-digit BitCode
            signalLength = (bitTime*nbits*sampleRate) + (gapTime*(nbits-1)*sampleRate);
            bitInd       = zeros(1,nbits-1);
            for j = 1:nbits-1
                bitInd(j) = j * (bitTime+gapTime) * sampleRate;
            end
            
            bitInd    = bitInd + (bitTime*sampleRate / 5); % Shift bitInds forward for more reliability
            trialNums = cell(length(digSignal), 1);
            
            for i = 1:length(digSignal) % For each trial
                thisTrial = cell2mat(digSignal(i,1));
                if any(thisTrial == 1)
                    % Find trial start bit
                    oneInds     = find(thisTrial == 1);
                    signalStart = oneInds(1);
                    if ~(signalStart+signalLength > length(thisTrial)) % Entire BitCode must be recorded
                        % Extract BitCode
                        bitSignal    = thisTrial(signalStart:signalStart+signalLength, 1);
                        
                        %    figure; plot(digSignal); hold on
                        %    plot(bitInd, repmat(5,size(bitInd)),'r*')
                        
                        % Read and save BitCode
                        bitCode      = bitSignal(round(bitInd))';
                        trialNums{i} = binvec2dec(bitCode);
                        
                        % As QC, try to fill in missing trialNums
                        if i>4 & isempty(trialNums{i-3}) & (trialNums{i} - trialNums{i-4}) == 4 % 3 consecutive trials missing
                            trialNums{i-1} = trialNums{i} - 1;
                            trialNums{i-2} = trialNums{i} - 2;
                            trialNums{i-3} = trialNums{i} - 3;
                        elseif i>3 & isempty(trialNums{i-2}) & (trialNums{i} - trialNums{i-3}) == 3 % 2 consecutive trials missing
                            trialNums{i-1} = trialNums{i} - 1;
                            trialNums{i-2} = trialNums{i} - 2;
                        elseif i>2 & isempty(trialNums{i-1}) & (trialNums{i} - trialNums{i-2}) == 2 % 1 trial missing
                            trialNums{i-1} = trialNums{i} - 1;
                        end
                    end
                end
            end
        end 
        
        
        function wspurge (wsdata, obj)
            % Removes non-task video data at end of session
            % RD fall 2019
            
            [csvData] = obj.GetTable('DLCData');
            csvEnd    = height(csvData);
            wsEnd     = height(wsdata.wsTable);
            
            if csvEnd > wsEnd
                csvData(end-(csvEnd-wsEnd)+1:end, :) = [];
            end
            
            obj.RemoveTable('DLCData')
            obj.SetTable('DLCData', csvData, 'timeSeries')
            
        end
        
                
        function wsdata2se (wavesurferData, obj)
            % RD summer 2019
            
            % Load WaveSurfer data
            wsData      = wavesurferData.wsTable;
            wsTrialNums = wsData.trialNo;
                                                    
%             varNames = 
            % % Make table
%           [tb, preTb] = MSessionExplorer.MakeTimeSeriesTable(wavesurferData.wsTable.timeInMs{}, wavesurferData,...
%            'variableNames', wavesurferData.wsInfo.header.AllChannelNames);

%             se.userData.preTaskData.wavesurferData = preTb;
            obj.userData.wsInfo = wavesurferData.wsInfo;
            obj.SetTable('waveSurferData', wsData, 'timeSeries');
           
        end
        
        
        function bControlData = purge (bControlData, wavesurferData)
            % Removes unrecorded trials at start of session from BControl
            % data
            % RD summer 2019
            
            % Get trial number of first recorded trial
            startTrial     = cell2mat(wavesurferData.wsTable.trialNo(1));
            nMissingTrials = startTrial - cell2mat(bControlData.trialNums(1));
            
            % If there are unrecorded trials at start of session
            if nMissingTrials > 1
                bControlData.trialNums     = bControlData.trialNums(nMissingTrials+1:end);
                bControlData.blockType     = bControlData.blockType(nMissingTrials+1:end);
                bControlData.trialType     = bControlData.trialType(nMissingTrials+1:end);
                bControlData.trialResponse = bControlData.trialResponse(nMissingTrials+1:end);
                bControlData.visStimType   = bControlData.visStimType(nMissingTrials+1:end);
                bControlData.somStimType   = bControlData.somStimType(nMissingTrials+1:end);
            end
            
        end
        
        
        function loadbctdata (bctData, obj)
             % Import trial numbers from Bcontrol data
            bctNums        = bctData.trialNums;
            bctBlockType   = bctData.blockType;
            bctTrialType   = bctData.trialType;
            bctVisStimType = bctData.visStimType;
            bctSomStimType = bctData.somStimType;
            
            % Convert Trial response from double to cell
            bctTrialResponse = num2cell(bctData.trialResponse);
            
           
            % Create a table for behavior values
            varNames = {'trialNum', 'blockType', 'trialType', 'response', 'visStimType', 'somStimType'};
            bevTable = table(bctNums, bctBlockType, bctTrialType, bctTrialResponse, bctVisStimType, bctSomStimType,...
                'VariableNames',varNames);
                                   
            % Save to SE
            %     se.userData.satInfo = satStruct.info;
            %     se.SetTable('behavTime', satStruct.timeTable, 'eventTimes', satStruct.episodeTimeRef);
            %     se.SetTable('behavValue', satStruct.valueTable, 'eventValues');
            obj.SetTable('behavValue', bevTable, 'eventValues');
            
        end
        
        
        % Processing
        
        
        function removeJumps (obj, method)
            
            [dlcTraces] = obj.GetTable('DLCData');
            
            switch method
                
                case 'makenan'
                    for i=2:width(dlcTraces)
                        shifts        = cellfun(@diff, table2array(dlcTraces(:, i)), 'Uni', false);
                        largeShiftsUp = cellfun(@(x) find(x >= 5), shifts, 'Uni', false); % shifts larger than +5 pixels
                        largeShiftsDn = cellfun(@(x) find(x <= -5), shifts, 'Uni', false); % shifts larger than -5 pixels
                        shiftWinds    = cellfun(@(x,y) x-y, largeShiftsUp, largeShiftsDn, 'Uni', false);
                        % dlcTraces(:, i)(shiftWinds) = NaN;
                    end
                    
                case 'medfilt'
                    for i=2:width(dlcTraces)
                        dlcTraces(:, i) = cellfun(@(x) medfilt1(x, 5), table2array(dlcTraces(:, i)), 'Uni', false);
                    end
            end
            
            % Update dlcdata object
            obj.SetTable('DLCData', dlcTraces, 'timeSeries');
            
        end
        
        
        function getEventTimes (obj)
            
            disp ('Fetching event times');
            
            % Get data and session information from se
            
            totalTrials      = obj.numEpochs;
            triggerDelay     = 384;
                        
            wsData  = obj.GetTable('waveSurferData');
            time    = wsData.time;
            visStim = wsData.visLED;
            somStim = wsData.pz;
            lLick   = wsData.lickprt2;
            rLick   = wsData.lickprt1;
            
            % Get visual onsets and offsets in each trial
            for k = 1 : length(visStim)
                visInTrial = visStim{k};
                timeInTrial = time{k};
                visOnsetInds{k,1} = find(visInTrial > 1, 1);
                % the first one that crosses the threshold of visual stimulus
                visOffsetInds{k,1} = find(visInTrial > 1, 1, 'last');
                % the last one that crosses the threshold of visual stimulus
                if isempty(visOnsetInds{k,1}) == 1
                    visOnsetTimes{k,1} = NaN;
                else
                    visOnsetTimes{k,1} = timeInTrial(visOnsetInds{k,1});
                end
                if isempty(visOffsetInds{k,1}) == 1
                    visOffsetTimes{k,1} = NaN;
                else
                    visOffsetTimes{k,1} = timeInTrial(visOffsetInds{k,1});
                end
            end
            
            % Get som onsets and offsets in each trial
            for k = 1 : length(somStim)
                somInTrial = somStim{k};
                timeInTrial = time{k};
                somOnsetInds{k,1} = find(somInTrial > 2, 1);
                somOffsetInds{k,1} = find(somInTrial > 2, 1, 'last');
                if isempty(somOnsetInds{k,1}) == 1
                    somOnsetTimes{k,1} = NaN;
                else
                    somOnsetTimes{k,1} = timeInTrial(somOnsetInds{k,1});
                end
                
                if isempty(somOffsetInds{k,1}) == 1
                    somOffsetTimes{k,1} = NaN;
                else
                    somOffsetTimes{k,1} = timeInTrial(somOffsetInds{k,1});
                end
            end
            
            % Get right licks in each trial
            for k = 1 : length(rLick)
                rLickInTrial = rLick{k};
                timeInTrial = time{k};
                rLickInds{k,1} = find(rLickInTrial > 1);
                rLickOnsetInds{k,1} = find(rLickInTrial > 1, 1);
                
                if isempty(rLickInds{k,1}) == 1
                    rLickTimes{k,1} = NaN;
                else
                    rLickTimes{k,1} = timeInTrial(rLickInds{k,1});
                end
                
                if isempty(rLickOnsetInds{k,1}) == 1
                    rLickOnsetTimes{k,1} = NaN;
                else
                    rLickOnsetTimes{k,1} = timeInTrial(rLickOnsetInds{k,1});
                end
            end
            
            % Get left licks in each trial
            for k = 1 : length(lLick)
                lLickInTrial = lLick{k};
                timeInTrial = time{k};
                lLickInds{k,1} = find(lLickInTrial > 1);
                lLickOnsetInds{k,1} = find(lLickInTrial > 1, 1);
                
                if isempty(lLickInds{k,1}) == 1
                    lLickTimes{k,1} = NaN;
                else
                    lLickTimes{k,1} = timeInTrial(lLickInds{k,1});
                end
                
                if isempty(lLickOnsetInds{k,1}) == 1
                    lLickOnsetTimes{k,1} = NaN;
                else
                    lLickOnsetTimes{k,1} = timeInTrial(lLickOnsetInds{k,1});
                end
            end
            
            % Create combined stim onset table
            stimOnset = {};
            for k = 1 : length(somOnsetTimes)
                if isequaln(somOnsetTimes{k,1},NaN) == 0 && isequaln(visOnsetTimes{k,1}, NaN) == 1
                    stimOnset{k, 1} = somOnsetTimes{k,1};
                elseif isequaln(somOnsetTimes{k,1}, NaN) == 1 && isequaln(visOnsetTimes{k,1}, NaN) == 0
                    stimOnset{k, 1} = visOnsetTimes{k,1};
                else
                    stimOnset{k, 1} = NaN;
                end
            end
            
            % Create stim onset copy without NaNs
            stimOnsetNaNZeros            = cell2mat(stimOnset);
            nanTrials                    = isnan(stimOnsetNaNZeros);
            stimOnsetNaNZeros(nanTrials) = 500; % ms
            
            % Create behavTime table
            varNames = {'somOnset', 'somOffset', 'visOnset', 'visOffset', 'stimOnset'...
                'rLickOnset', 'lLickOnset', 'rLick', 'lLick','stimOnsetNaNZeros'};
            behavTime = table(somOnsetTimes, somOffsetTimes, visOnsetTimes, visOffsetTimes, stimOnset,...
                rLickOnsetTimes, lLickOnsetTimes, rLickTimes, lLickTimes, stimOnsetNaNZeros,...
                'VariableNames',varNames);
            
            
            % Add behavTime to dlcdata
            obj.SetTable('behavTime', behavTime, 'eventTimes');
        end
        
        
        function ttMap (obj)
            % Adapted
            % RD summer 2019
            
            bv = obj.GetTable('behavValue');
            
            % regular contigency: T-lick Right, V-lick Left
            TTTind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'Stim_Som_NoCue') & ...
                ismember(cell2mat(bv.response), [1]) ;
            
            TTVind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'Stim_Som_NoCue') & ...
                ismember(cell2mat(bv.response), [2]) ;
            
            TTNind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'Stim_Som_NoCue') & ...
                ismember(cell2mat(bv.response), [0]) ;
            
            TVTind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'Stim_Vis_NoCue') & ...
                ismember(cell2mat(bv.response), [1]) ;
            
            TVVind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'Stim_Vis_NoCue') & ...
                ismember(cell2mat(bv.response), [2]) ;
            
            TVNind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'Stim_Vis_NoCue') & ...
                ismember(cell2mat(bv.response), [0]) ;
            
            VTTind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'Stim_Som_NoCue') & ...
                ismember(cell2mat(bv.response), [1]) ;
            
            VTVind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'Stim_Som_NoCue') & ...
                ismember(cell2mat(bv.response), [2]) ;
            
            VTNind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'Stim_Som_NoCue') & ...
                ismember(cell2mat(bv.response), [0]) ;
            
            VVTind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'Stim_Vis_NoCue') & ...
                ismember(cell2mat(bv.response), [1]) ;
            
            VVVind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'Stim_Vis_NoCue') & ...
                ismember(cell2mat(bv.response), [2]) ;
            
            VVNind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'Stim_Vis_NoCue') & ...
                ismember(cell2mat(bv.response), [0]) ;
            
            TCTind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'catch') & ...
                ismember(cell2mat(bv.response), [1]) ;
            
            TCVind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'catch') & ...
                ismember(cell2mat(bv.response), [2]) ;
            
            TCNind = ismember(bv.blockType, 'Whisker') & ismember(bv.trialType, 'catch') & ...
                ismember(cell2mat(bv.response), [0]) ;
            
            VCTind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'catch') & ...
                ismember(cell2mat(bv.response), [1]) ;
            
            VCVind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'catch') & ...
                ismember(cell2mat(bv.response), [2]) ;
            
            VCNind = ismember(bv.blockType, 'Visual') & ismember(bv.trialType, 'catch') & ...
                ismember(cell2mat(bv.response), [0]) ;
            
            varNames = {'TTT','TTV', 'TTN', 'TVT', 'TVV', 'TVN', 'VTT','VTV', 'VTN', 'VVT', 'VVV', 'VVN',...
                'TCT', 'TCV', 'TCN', 'VCT', 'VCV', 'VCN'};
            
            ttMap = table(TTTind,TTVind, TTNind, TVTind, TVVind, TVNind, VTTind,VTVind, VTNind, VVTind, VVVind, VVNind, ...
                TCTind, TCVind, TCNind, VCTind, VCVind, VCNind, 'VariableNames',varNames); % 18 trial types
            
%             varNames = {'TCT', 'TCV', 'TCN', 'VCT', 'VCV', 'VCN'};
%             
%             ttMap = table(TCTind, TCVind, TCNind, VCTind, VCVind, VCNind, 'VariableNames',varNames); % 18 trial types
            
            % Add ttMap to behavValue table
            newBehavValue = [bv, ttMap];
            obj.SetTable('behavValue', newBehavValue);
                        
        end
        
        
        function findAngles (obj)
            % Find whisker angles
            
            disp ('Calculating whisker angles')
            
            [dlcTraces] = obj.GetTable('DLCData');
            
            tarAngle = cell(height(dlcTraces), 1);
            surAngle = tarAngle; 
            
            for i=1:height(dlcTraces)
                % Calculate angle at target whisker base
                tarBase     = dlcTraces.tarBase_x{i} - dlcTraces.tarEnd_x{i};
                tarPerpend  = dlcTraces.tarEnd_y{i} - dlcTraces.tarBase_y{i};
                tarAngle{i} = atan(tarBase ./ tarPerpend) * (180/pi);
                
                % Calculate angle of surrogate whisker 
                surBase     = dlcTraces.surMid_x{i} - dlcTraces.surBase_x{i};
                surPerpend  = dlcTraces.surMid_y{i} - dlcTraces.surBase_y{i};
                surAngle{i} = atan(surBase ./ surPerpend) * (180/pi);
            end
            
            dlcTraces                                 = [dlcTraces, tarAngle, surAngle];
            dlcTraces.Properties.VariableNames{end-1} = 'tarAngle';
            dlcTraces.Properties.VariableNames{end}   = 'surAngle';
            obj.SetTable('DLCData', dlcTraces);
            
        end
        
        
        function baselineAngles (obj, window, zerobin)
            % Baseline data by mean of specified pre-stimulus window
            % RD fall 2020
            
            framerate = obj.userData.framerate;
            if nargin < 3
                zerobin = 1222;
            end
            if nargin < 2
                window = 100;
            end
            
            baseFrames = round(window * (framerate / 1000));
            baseWind   = (zerobin - baseFrames : zerobin - 1);
            
            [dlcTraces] = obj.GetTable('DLCData');
            
            [tarAngBase, surAngBase] = deal(cell(height(dlcTraces), 1));
                        
            for i=1:height(dlcTraces)
                % Baseline-subtract whisker angles
                try
                    tarAngBase{i} = dlcTraces.tarAngle{i} - nanmean(dlcTraces.tarAngle{i}(baseWind, 1));
                    surAngBase{i} = dlcTraces.surAngle{i} - nanmean(dlcTraces.surAngle{i}(baseWind, 1));
                catch
                    tarAngBase{i} = dlcTraces.tarAngle{i} - nanmean(dlcTraces.tarAngle{i}(:, 1));
                    surAngBase{i} = dlcTraces.surAngle{i} - nanmean(dlcTraces.surAngle{i}(:, 1));
                end
            end
            
            dlcTraces                                 = [dlcTraces, tarAngBase, surAngBase];
            dlcTraces.Properties.VariableNames{end-1} = 'tarAngBase';
            dlcTraces.Properties.VariableNames{end}   = 'surAngBase';
            obj.SetTable('DLCData', dlcTraces);
            
        end
        
        
        % Analysis
        
        
        function trialGroups = groupTrials (obj)
            
            disp ('Grouping trials');
            
            % Get trial-type indices
            trialTypes = {'TTT', 'TTV', 'TTN', 'TVT', 'TVV', 'TVN', 'VTT', 'VTV', 'VTN', 'VVT', 'VVV',...
                'VVN', 'TCT', 'TCV', 'TCN', 'VCT', 'VCV', 'VCN'};
            ttData     = array2table(obj.GetColumn('behavValue', trialTypes), 'VariableNames', trialTypes);
            
            [dlcdata] = obj.GetTable ('DLCData');
                                   
            % Construct trial group indices
            respTypes                       = {'Tac', 'Vis', 'Lick', 'NoLick', 'Touch', 'Light',...
                'TacLick', 'TacNoLck', 'VisLick', 'VisNoLck'};
            temp                            = ttData;
            ttData                          = array2table(zeros(height(temp), 10));
            ttData.Properties.VariableNames = respTypes;
            ttData.Tac                      = logical(temp.TTT + temp.TTV + temp.TTN + temp.VTT + temp.VTV + temp.VTN);
            ttData.Vis                      = logical(temp.TVT + temp.TVV + temp.TVN + temp.VVT + temp.VVV + temp.VVN);
            ttData.Lick                     = logical(temp.TTT + temp.VVV + temp.TTV + temp.TVT + temp.TVV + temp.VTT...
                                                        + temp.VVT);
            ttData.NoLick                   = logical(temp.TTN + temp.VTN + temp.VVN + temp.TVN);                                                                
%             ttData.Lick                     = logical(temp.TTT + temp.VVV);
%             ttData.NoLick                   = logical(temp.TTV + temp.TTN + temp.TVT + temp.TVV + temp.VTT + temp.VTN...
%                  + temp.VVT + temp.VVN);
            ttData.Touch                    = logical(temp.TTT + temp.TTV + temp.TTN + temp.TVT  + temp.TVV + temp.TVN...
                 + temp.TCT + temp.TCV + temp.TCN);
            ttData.Light                    = logical(temp.VTT + temp.VTV + temp.VTN + temp.VVT + temp.VVV + temp.VVN...
                 + temp.VCT + temp.VCV + temp.VCN);
            ttData.TacLick                  = logical(temp.TTT + temp.TTV + temp.VTT + temp.VTV);
            ttData.TacNoLck                 = logical(temp.TTN + temp.VTN);
            ttData.VisLick                  = logical(temp.TVT + temp.TVV + temp.VVT + temp.VVV);
            ttData.VisNoLck                 = logical(temp.TVN + temp.VVN);
            clear temp;
            
            % Group trials
            trialGroups = cell(1, size(ttData, 2));
            for tt = 1:size(ttData, 2) % For each trial group
                trialGroups{1, tt} = dlcdata(ttData {:,tt}, 2:end);
            end
            
            % Create a table
            trialGroups = cell2table(trialGroups, 'VariableNames', respTypes);
            
        end
        
        
        function ttData = getTrialMap (obj)
            % RD spring 2023
            
            % Get trial-type indices
            trialTypes = {'TTT', 'TTV', 'TTN', 'TVT', 'TVV', 'TVN', 'VTT', 'VTV', 'VTN', 'VVT', 'VVV',...
                'VVN', 'TCT', 'TCV', 'TCN', 'VCT', 'VCV', 'VCN'};
            ttData     = array2table(obj.GetColumn('behavValue', trialTypes), 'VariableNames', trialTypes);
            
            
            trialTypes_b = {'Tac', 'Vis', 'TacHit', 'TacFA', 'TacMiss', 'TacCR', ...
                'VisHit', 'VisFA', 'VisMiss', 'VisCR'};
            temp                            = ttData;
            ttData                          = array2table(zeros(height(temp), 10));
            ttData.Properties.VariableNames = trialTypes_b;
            ttData.Tac                      = logical(temp.TTV + temp.TVT + temp.TVV + temp.TTT + temp.TTN + temp.TVN);
            ttData.Vis                      = logical(temp.VTV + temp.VVT + temp.VTT + temp.VVV + temp.VVN + temp.VTN);
            ttData.TacHit                   = temp.TTT;
            ttData.TacFA                    = logical(temp.TTV + temp.TVT + temp.TVV);
            ttData.TacMiss                  = temp.TTN;
            ttData.TacCR                    = temp.TVN;
            ttData.VisHit                   = temp.VVV;
            ttData.VisFA                    = logical(temp.VTV + temp.VVT + temp.VTT);
            ttData.VisMiss                  = temp.VVN;
            ttData.VisCR                    = temp.VTN;
            clear temp;
            
        end
                
        
        function trialAverages = averages (obj)
            % RD summer 2019
            
            disp ('Computing trial averages');
            
            % Get trial-type indices
            ttData = obj.getTrialMap (obj);
            trialTypes_b = {'Tac', 'Vis', 'TacHit', 'TacFA', 'TacMiss', 'TacCR', ...
                'VisHit', 'VisFA', 'VisMiss', 'VisCR'};
            
            % Get data
            angles    = {'tarAngle', 'surAngle'};
            [dlcData] = obj.GetColumn ('DLCData', angles);
            dlcData   = cell2table(dlcData);
            
            % Compute average(s)
            avgData = cell(5, size(ttData, 2));
            
            for tt = 1:size(ttData, 2) % compute for each trial type
                ttInd   = ttData {:,tt}; % curly brackets to make a logical array
                
                mean_tt  = [];
                std_tt   = [];
                sem_tt   = [];
                lbSem_tt = [];
                hbSem_tt = [];
                
                for i = 1:width (dlcData) % compute for each individual angle for this trial type
                    angleData = dlcData(:,i);
                    angleData = angleData {ttInd,1};
                    angle_tb  = NaN(numel(angleData), max(cellfun(@length, angleData)));
                    
                    for trial = 1 : length(angleData)
                        if ~any(isnan(angleData{trial}))
                            angle_tb (trial, 1:length(angleData{trial})) = [angleData{trial}];
                        end
                    end
                    
                    mean_angle_tt  = nanmean(angle_tb',2); % mean trace for this trial type and this angle
                    std_angle_tt   = nanstd(angle_tb',0,2); % std trace for this trial type and this angle
                    sem_angle_tt   = std_angle_tt / sqrt(length(angle_tb));
                    lbSem_angle_tt = mean_angle_tt - sem_angle_tt;
                    hbSem_angle_tt = mean_angle_tt + sem_angle_tt;
                    
                    % Add each cell trace to a table
                    mean_tt  = [mean_tt, mean_angle_tt];
                    std_tt   = [std_tt, std_angle_tt];
                    sem_tt   = [sem_tt, sem_angle_tt];
                    lbSem_tt = [lbSem_tt, lbSem_angle_tt];
                    hbSem_tt = [hbSem_tt, hbSem_angle_tt];
                end
                
                avgData{1, tt} = mean_tt;
                avgData{2, tt} = std_tt;
                avgData{3, tt} = sem_tt;
                avgData{4, tt} = lbSem_tt;
                avgData{5, tt} = hbSem_tt;
            end
            
            % Create a table
            rowNames = {'mean','std', 'sem','lbSem', 'hbSem'};
            trialAverages = cell2table(avgData, 'VariableNames', trialTypes_b, 'RowNames', rowNames);
        end
        
        
        function avgData = getAvgs (varargin)
            % Returns avg whisker data for same trial types as roc
            %
            % RD fall 2020
                       
            % Handle user inputs
            p = inputParser();
            p.addRequired ('trialGroups')
            p.addParameter ('framerate', 500.15, @isnumeric)
            p.addParameter ('ttPairInds', [1:10, 7, 9], @isnumeric)
            p.addParameter ('zerobin', 1222, @isnumeric)
            
            p.parse(varargin{:});
            trialGroups = p.Results.trialGroups;
            framerate   = p.Results.framerate;
            ttPairInds  = p.Results.ttPairInds;
            zerobin     = p.Results.zerobin;
            
%             disp ('Doing ideal observer analysis. This might take a while')
            warning ('off','stats:perfcurve:SubSampleWithMissingClasses') % Suppress subsampling warning
                        
            % Set durations
            analyzeWind   = 2500; % Number of frames to analyze per trial; roughly first 5 secs at 500.15 fps
            timeBin       = 25; % ms
%             framesPerBin  = round(timeBin * (framerate / 1000));
            framesPerBin  = [floor(timeBin * (framerate / 1000)), ceil(timeBin * (framerate / 1000))];
            numBins       = floor(analyzeWind / mean(framesPerBin(1)));
            
            % Initialize cell array for avg data
            avgData = cell(12, 5);
                        
            for i = 1:6 % Five pairs of trial types
                ttIndices   = ttPairInds(1, [i*2-1 i*2]);
                trialData   = table2array(trialGroups (1, ttIndices(1))); % Get data for first trial type
                trialData1 = table2array(trialData{1,1}(:, end-1:end)); % Get only whisker angle data
%                 trialData1b = table2array(trialData{1,1}(:, end-1:end)); % Get only baselined whisker angle data
                trialData   = table2array(trialGroups (1, ttIndices(2))); % Get data for second trial type
                trialData2 = table2array(trialData{1,1}(:, end-1:end));
%                 trialData2b = table2array(trialData{1,1}(:, end-1:end));
                %                 minLength  = min(cellfun(@length, vertcat(trialData1, trialData2)));
                                                
                                             
                % Fill missing data in with NaNs
                trialData1 = dlcdata.fillNans (trialData1, analyzeWind);
                trialData2 = dlcdata.fillNans (trialData2, analyzeWind);
                
                [avgTar1, avgTar2, avgSur1, avgSur2, semTarUp1, semTarUp2,...
                    semTarDn1, semTarDn2, semSurUp1, semSurUp2, semSurDn1, semSurDn2]...
                    = deal(nan(1,numBins));
                
                % ROC analysis for each timebin
                for j = 1:numBins 
%                     if mod(j, 2)
                        nframes = framesPerBin(1);
%                     else
%                         nframes = framesPerBin(2);
%                     end
                    % Get start and end frames for this bin
                    binStart = (nframes * (j-1)) +1;   
                    binEnd   = nframes * j;
                    % Extract bin means for each trial type
                    
                    wrapper = @(x) nanmean(x(binStart:binEnd));
                    if j == numBins
                        wrapper = @(x) nanmean(x(binStart:end));
                    end
                    tt1Tar  = cellfun(wrapper, trialData1(:, 1));
                    tt2Tar  = cellfun(wrapper, trialData2(:, 1));
                    tt1Sur  = cellfun(wrapper, trialData1(:, 2));
                    tt2Sur  = cellfun(wrapper, trialData2(:, 2));
                    
                    % Get means
                    [avgTar1(j), semTarUp1(j), semTarDn1(j)] = dlcdata.meansem(tt1Tar);
                    [avgTar2(j), semTarUp2(j), semTarDn2(j)] = dlcdata.meansem(tt2Tar);
                    [avgSur1(j), semSurUp1(j), semSurDn1(j)] = dlcdata.meansem(tt1Sur);
                    [avgSur2(j), semSurUp2(j), semSurDn2(j)] = dlcdata.meansem(tt2Sur);
                    disp(['Bin ', num2str(j), ' done'])
                end
                avgData (1, i)  = {avgTar1};
                avgData (2, i)  = {semTarUp1};
                avgData (3, i)  = {semTarDn1};
                avgData (4, i)  = {avgTar2};
                avgData (5, i)  = {semTarUp2};
                avgData (6, i)  = {semTarDn2};
                avgData (7, i)  = {avgSur1};
                avgData (8, i)  = {semSurUp1};
                avgData (9, i)  = {semSurDn1};
                avgData (10, i) = {avgSur2};
                avgData (11, i) = {semSurUp2};
                avgData (12, i) = {semSurDn2};
                
                disp(['Iteration ', num2str(i), ' done'])
            end
            
            % Set tables
            ttNames    = {'Tactile_Visual', 'Lick_NoLick', 'Touch_Light', 'TouchHit_TouchMiss',...
                'VisHit_Vis_Miss', 'TouchHit_VisHit'};
            rowNames_b = {'Target_mean1', 'Target_SEMUp1', 'Target_SEMDn1',...
                          'Target_mean2', 'Target_SEMUp2', 'Target_SEMDn2'...
                          'Surrogate_mean1', 'Surrogate_SEMUp1', 'Surrogate_SEMDn1'...
                          'Surrogate_mean2', 'Surrogate_SEMUp2', 'Surrogate_SEMDn2'}; 
            
            avgData = cell2table(avgData, 'VariableNames', ttNames, 'RowNames', rowNames_b);
            
        end
                  
        
        function windAvg = extractTrialWind (obj, col, window, mask, zerobin)
            % Returns average of frames specified in window from columns
            % specified in col and all trials
            %
            % RD fall 2020
            
            if nargin < 5
                zerobin = 1222;
            end
            if nargin < 4
                mask = [];
            end
            
            % Get data
            framerate = obj.userData.framerate;
            [data]    = obj.GetTable('DLCData');
            data      = data{:, col};
            
            % Select correct trial type
            if ~isempty (mask)
                data = data(mask, 1);
            end
            
            % Set frame numbers
            frames = round(window * (framerate / 1000)) + zerobin;
            
            % Extract
            trialMask = cellfun(@length, data) >= max(frames); % Only consider trials that are long enough
            windAvg   = cellfun(@(x) nanmean(x(frames, 1)), data(trialMask, :), 'Uni', 0);
            
        end
             
        
        function [windDiffs, windAUC] = getWindScatter (stim, whisk, stimWind, preWind)
            % Returns pre-stimulus baseline difference between TacLick and TacNoLick vs
            % stimulus peak difference/AUC between the same trial-types
            %
            % RD fall 2020
            
            mainPath  = 'D:\data\project_rcm\finkel et al\matlab\dd\';
            sesData   = dlcdata.readSesData(mainPath);
            windDiffs = nan(height(sesData), 4);
            windAUC   = nan(height(sesData), 5);
            mouse     = 1;
            
            % Set windows
            if nargin < 4
                preWind  = [-100 -1];
            end    
            if nargin < 3
                stimWind = [14 34];
            end
            
            for i = 1:height(sesData)
                
                % Load dd file
                if ~isempty(cell2mat(sesData{i, 1}))
                    thisMouse   = cell2mat(sesData{i, 1});
                    thisSession = num2str(sesData{i, 2});
                else
                    continue
                end
                sessionName = [thisMouse, '_', thisSession];
                ddfile      = [mainPath, sessionName, '_dlcdata.mat'];
                try
                    load(ddfile);
                catch
                    error ('Could not load dlcdata object')
                end
                
                % Get trial masks
%                 trialTypes = {'TTT', 'TTV', 'VTT', 'VTV', 'VTN', 'TTN'};
%                 ttData     = array2table(dd.GetColumn('behavValue', trialTypes), 'VariableNames', trialTypes);
%                 tacLick    = logical(ttData.TTT + ttData.TTV + ttData.VTT + ttData.VTV);
%                 tacNoLck   = logical(ttData.TTN + ttData.VTN);
                if strcmp(stim, 'tac')
                    [Lick, NoLck, ~,~] = dlcdata.quickTTMap (dd);
                elseif strcmp(stim, 'vis')
                    [~,~, Lick, NoLck] = dlcdata.quickTTMap (dd);
                end
                
                % Set whisker values
                if strcmp(whisk, 'tar')
                    raw = 16;
                    base = 18;
                elseif strcmp(whisk, 'sur')
                    raw = 17;
                    base = 19;
                end
                
                % Get avgs
                lckP   = cell2mat(dd.extractTrialWind(dd, raw, preWind, Lick)); % Raw data for pre-stim window
                noLckP = cell2mat(dd.extractTrialWind(dd, raw, preWind, NoLck));
                lckS   = cell2mat(dd.extractTrialWind(dd, base, stimWind, Lick)); % Baseline-subtracted data for stim
                noLckS = cell2mat(dd.extractTrialWind(dd, base, stimWind, NoLck));
                
                % Save diffs
                windDiffs(i, 1) = mean(noLckP) - mean(lckP);
                windDiffs(i, 2) = mean(noLckS) - mean(lckS);
                
                % Save auc significance flag
                [auc, quickauc] = dd.issig(lckS, noLckS, 1000, height(sesData));
                windAUC(i, 1:3) = auc;
                [windAUC(i, 4), windDiffs(i, 3)] = deal(quickauc);
                
                % Save mouse id
                if i > 1
                    if ~strcmp(thisMouse, prevMouse)
                        mouse = mouse + 1;
                    end
                end
                [windAUC(i, 5), windDiffs(i, 4)] = deal(mouse);
                prevMouse = thisMouse;
                
            end
        end
        
        
        % Trial rates
        
        
        function rollingRate = getRollingRate (ar, windSize, stepSize)
            % RD spring 2023
      
            % Init storage
            rollingRate = zeros(floor(numel(ar) / stepSize), 1);
            for i = 1 : stepSize : numel(ar)
                % For earlier bins, window will start at start of session
                windStart = max(1, i - windSize);
                window = ar(windStart : i);
                rollingRate(i) = sum(window) / numel(window);
            end
            
        end
        
                
        function ttRates = computeTrialRates (obj, windSize, stepSize)
            % RD spring 2023
            
            % Set defaults
            if nargin < 3
                stepSize = 1;
            end
            % Use window size of 50 by default
            if nargin < 2
                windSize = 50;
            end
                
            ttData = obj.getTrialMap (obj);
            Hit = logical(ttData.TacHit) | logical(ttData.VisHit);
            Miss = logical(ttData.TacMiss) | logical(ttData.VisMiss);
            MaxHit = Hit | Miss;
            CR = logical(ttData.TacCR) | logical(ttData.VisCR);
            FA = logical(ttData.TacFA) | logical(ttData.VisFA);
            MaxCR = CR | FA;
            
            allTrials = Hit | Miss | CR | FA;
            % Sanity check
            assert (~any(allTrials > 1), 'Incorrect trial inds')
            
            % Rolling window calculations
            % Go trials
            maxHitRate = obj.getRollingRate(MaxHit, windSize, stepSize);
            Hit = obj.getRollingRate(Hit, windSize, stepSize) ./ maxHitRate;
            Miss = obj.getRollingRate(Miss, windSize, stepSize) ./ maxHitRate;
            % No-go trials
            maxCRRate = obj.getRollingRate(MaxCR, windSize, stepSize);
            CR = obj.getRollingRate(CR, windSize, stepSize) ./ maxCRRate;
            FA = obj.getRollingRate(FA, windSize, stepSize) ./ maxCRRate;
            ttRates = table(Hit, Miss, CR, FA, 'variableNames', {'Hit', 'Miss', 'CR', 'FA'});
            
        end
        
        
        function plotTTRates (Hit, Miss, CR, FA, skip)
            % RD spring 2023
            
            % Set default number of trials to skip
            if nargin < 5
                skip = 5;
            end
            figure
            hold on
            % Plot while leaving out first few 
            dlcdata.plotWithErrors(Hit(skip+1:end, :), 'b', 'b');
            dlcdata.plotWithErrors(Miss(skip+1:end, :), 'k', 'k');
            dlcdata.plotWithErrors(CR(skip+1:end, :), "#A2142F", 'r');
            dlcdata.plotWithErrors(FA(skip+1:end, :), "#77AC30", 'g');
            
            t_x = 150;
            t_y = 0.95;
            t_size = 15;
            t = text (t_x, t_y, 'Hit');
            set (t, 'Color', 'b', 'Fontsize', t_size)
            t = text (t_x, t_y - 0.06, 'Miss');
            set (t, 'Color', 'k', 'Fontsize', t_size)
            t = text (t_x + 30, t_y, 'CR');
            set (t, 'Color', 'r', 'Fontsize', t_size)
            t = text (t_x + 30, t_y - 0.06, 'FA');
            set (t, 'Color', 'g', 'Fontsize', t_size)
            
            xlabel ('Trial no.')%, 'FontWeight', 'bold')
            ylabel ('Fraction of trials')
            xticks ([100, 200])
            yticks (linspace(0, 1, 6))
                
            ax = gca;
            ax.FontSize = 15;
            set(ax, 'TickDir', 'out')
        end
        
        
        % Reaction times
        
        
        function [tacRT, visRT] = getReactionTimes (mainPath)
            % Compiles reaction times for all Hit trials across sessions
            %
            % RD fall 2020
            
            if nargin < 1
                mainPath = 'E:\data\matlab\dd\';
            end
            
            sesData   = dlcdata.readSesData(mainPath);
            [tacRT, visRT] = deal([]);
            
            
            for i = 1:height(sesData)
                
                % Load dd file
                if ~isempty(cell2mat(sesData{i, 1}))
                    thisMouse   = cell2mat(sesData{i, 1});
                    thisSession = num2str(sesData{i, 2});
                else
                    continue
                end
                sessionName = [thisMouse, '_', thisSession];
                ddfile      = [mainPath, sessionName, '_dlcdata.mat'];
                try
                    load(ddfile);
                catch
                    error ('Could not load dlcdata object')
                end
                
                % Get trial masks
                trialTypes = {'TTT', 'TTV', 'VTT', 'VTV', 'TVT', 'TVV', 'VVT', 'VVV'};
                ttData     = array2table(dd.GetColumn('behavValue', trialTypes), 'VariableNames', trialTypes);
                tacLick    = logical(ttData.TTT); % + ttData.TTV + ttData.VTT + ttData.VTV);
                visLick    = logical(ttData.VVV); % + ttData.TVT + ttData.TVV + ttData.VVT);
                                
                % Get data
                colNames = {'pz', 'visLED', 'lickprt1', 'lickprt2'};
                wsData   = dd.GetColumn('waveSurferData', colNames);
                
                % Get tactile RTs
                tacdata        = wsData(tacLick, [1 3]); % Tactile-lick right scheme
                tacOnsets      = cell2mat(cellfun(@(x) find(x > 1.2, 1), tacdata(:,1), 'Uni', 0));
                tacTimes       = nan(length(tacdata), 2);
                tacTimes(:, 2) = ones(length(tacdata), 1) * i; % Save mouse id
                for t = 1:length(tacdata)
                    thisTrial   = tacdata{t, 2};
                    thisTrial   = thisTrial(tacOnsets(t)+1:end, 1); % Only consider post-stimulus onset period
                    tacTimes(t) = find(thisTrial > 1.5, 1) / 20; % in ms
                end
                
                % Get visual RTs
                visdata        = wsData(visLick, [2 4]); % Visual-lick left
                visOnsets      = cell2mat(cellfun(@(x) find(x > 1.5, 1), visdata(:,1), 'Uni', 0));
                visTimes       = nan(length(visdata), 2);
                visTimes(:, 2) = ones(length(visdata), 1) * i;
                for t = 1:length(visdata)
                    thisTrial   = visdata{t, 2};
                    thisTrial   = thisTrial(visOnsets(t)+1:end, 1); % Only consider post-stimulus onset period
                    visTimes(t) = find(thisTrial > 1.5, 1) / 20; % in ms
                end
                                          
                % Append
                tacRT = [tacRT; tacTimes];
                visRT = [visRT; visTimes];
                
            end
                        
        end
        
        
        function medRTs = getMedianRTs (rt)
            % Returns median RT values for each session
            % RD spring 21
            
            numses = numel(unique(rt(:,2)));
            medRTs = nan(numses, 1);
            
            for ses = 1:numses
                medRTs(ses) = median(rt(rt(:, 2)==ses, 1));
            end
            
        end
        
        
        % Ideal observer
        
        
        function [auROCData, btstrp_auROCData, stdROCData] = getauROCs (varargin)
            % Ideal observer analysis for whisker data. 'btstrp_auROCData'
            % contains mean and 95% confidence interval bounds of area
            % under ROC curve of performance by an ideal observer
            % discriminating between different trial types. Values are
            % calculated for every 25 ms timebin
            %
            % RD fall 2019
                       
            % Handle user inputs
            p = inputParser();
            p.addRequired ('trialGroups')
            p.addParameter ('method', 'buitin', @ischar)
            p.addParameter ('framerate', 500.15, @isnumeric)
            p.addParameter ('nboot', 1000, @isnumeric)
            p.addParameter ('ttPairInds', [1:10, 7, 9], @isnumeric)
            p.addParameter ('zerobin', 1222, @isnumeric)
            
            p.parse(varargin{:});
            trialGroups = p.Results.trialGroups;
            method      = p.Results.method;
            framerate   = p.Results.framerate;
            nboot       = p.Results.nboot;
            ttPairInds  = p.Results.ttPairInds;
            zerobin     = p.Results.zerobin;
            
            disp ('Doing ideal observer analysis. This might take a while')
            warning ('off','stats:perfcurve:SubSampleWithMissingClasses') % Suppress subsampling warning
                        
            % Set durations
            preStimDur    = 1000; % ms
            preStimFrames = round(preStimDur * (framerate / 1000));
            preStimWind   = (zerobin - preStimFrames : zerobin - 1);
            analyzeWind   = 2500; % Number of frames to analyze per trial; roughly first 5 secs at 500.15 fps
            timeBin       = 25; % ms
%             framesPerBin  = round(timeBin * (framerate / 1000));
            framesPerBin  = [floor(timeBin * (framerate / 1000)), ceil(timeBin * (framerate / 1000))];
            numBins       = floor(analyzeWind / mean(framesPerBin(1)));
            
            % Initialize cell arrays for auROC data
            auROCData        = cell(2, 6);
            btstrp_auROCData = cell(6, 6);
            stdROCData       = cell(6, 6);
            
            for i = 1:6 % Five pairs of trial types
                ttIndices   = ttPairInds(1, [i*2-1 i*2]);
                trialData   = table2array(trialGroups (1, ttIndices(1))); % Get data for first trial type
                trialData1  = table2array(trialData{1,1}(:, end-3:end-2)); % Get only whisker angle data
                trialData1b = table2array(trialData{1,1}(:, end-1:end)); % Get only baselined whisker angle data
                trialData   = table2array(trialGroups (1, ttIndices(2))); % Get data for second trial type
                trialData2  = table2array(trialData{1,1}(:, end-3:end-2));
                trialData2b = table2array(trialData{1,1}(:, end-1:end));
                %                 minLength  = min(cellfun(@length, vertcat(trialData1, trialData2)));
                                                
                % Create labels array with trial type names
                labels                                = cell(length(trialData1)+length(trialData2), 1);
                [labels{1:length(trialData1), 1}]     = deal(trialGroups.Properties.VariableNames{1, ttIndices(1)});
                [labels{length(trialData1)+1:end, 1}] = deal(trialGroups.Properties.VariableNames{1, ttIndices(2)});
                
                % Set posclass label
%                 if mean(cell2mat(trialData1(:,1))) > mean(cell2mat(trialData2(:,1)))
%                     posclass = labels{1};
%                 else
%                     posclass = labels{end};
%                 end
                posclass = labels{1};
                
                % Fill missing data in with NaNs
                trialData1  = dlcdata.fillNans (trialData1, analyzeWind);
                trialData2  = dlcdata.fillNans (trialData2, analyzeWind);
                trialData1b = dlcdata.fillNans (trialData1b, analyzeWind);
                trialData2b = dlcdata.fillNans (trialData2b, analyzeWind);
                
                
                % Generate auROC scores for standard deviation in pre-stim window
                % Extract pre-stim window
                tar1 = cellfun(@(x) x(preStimWind), trialData1(:,1), 'Uni', 0); 
                tar2 = cellfun(@(x) x(preStimWind), trialData2(:,1), 'Uni', 0);
                sur1 = cellfun(@(x) x(preStimWind), trialData1(:,2), 'Uni', 0);
                sur2 = cellfun(@(x) x(preStimWind), trialData2(:,2), 'Uni', 0);
                % Get standard deviation values
                stdTar1 = cellfun(@nanstd, tar1);
                stdTar2 = cellfun(@nanstd, tar2);
                stdSur1 = cellfun(@nanstd, sur1);
                stdSur2 = cellfun(@nanstd, sur2);
                % Generate scores for total data
                scoreTar = [stdTar1; stdTar2];
                scoreSur = [stdSur1; stdSur2];
                % Get AUROCs
                [~,~,~,stdAUCT] = perfcurve(labels, scoreTar, posclass, 'NBoot', nboot, 'Alpha', (0.05/14));
                [~,~,~,stdAUCS] = perfcurve(labels, scoreSur, posclass, 'NBoot', nboot, 'Alpha', (0.05/14));
                              
                [aucTar, aucSur, btaucTar, btaucSur, ciaucTarUp, ciaucTarDn, ciaucSurUp, ciaucSurDn]...
                    = deal(nan(1,numBins));
                % ROC analysis for each timebin
%                 for j = 1:numBins 
% %                     if mod(j, 2)
%                         nframes = framesPerBin(1);
% %                     else
% %                         nframes = framesPerBin(2);
% %                     end
%                     % Get start and end frames for this bin
%                     binStart = (nframes * (j-1)) +1;   
%                     binEnd   = nframes * j;
%                     % Extract bin means for each trial type
%                     
%                     wrapper = @(x) nanmean(x(binStart:binEnd));
%                     if j == numBins
%                         wrapper = @(x) nanmean(x(binStart:end));
%                     end
%                     tt1Tar  = cellfun(wrapper, trialData1b(:, 1));
%                     tt2Tar  = cellfun(wrapper, trialData2b(:, 1));
%                     tt1Sur  = cellfun(wrapper, trialData1b(:, 2));
%                     tt2Sur  = cellfun(wrapper, trialData2b(:, 2));
%                     
%                     % Generate scores for total data
%                     scoreTar = [tt1Tar; tt2Tar];
%                     scoreSur = [tt1Sur; tt2Sur];
%                     % Get AUROCs
%                     [AUCT, AUCS] = deal(NaN);
%                     if nargout > 2 % Only if absolute auROC values are requested
%                         [~,~,~,AUCT] = perfcurve(labels, scoreTar, posclass);
%                         [~,~,~,AUCS] = perfcurve(labels, scoreSur, posclass);
%                     end
%                                         
%                     % Create perfcurve function handle for bootstrap analysis
% %                     wrapper = @(x,y) perfcurve(x, y, posclass, 'ProcessNaN', 'ignore');
%                     
%                     % Generate scores for bootstrap sampling data
% %                     nboot      = length([tt1Tar; tt2Tar]);
%                     % Choose method for bootstrapping
%                     switch method
%                         
%                         case 'manual' % This method underestimates variance in bootstrap auROC values!
%                             btAUCT     = nan(1, nboot);
%                             btAUCS     = nan(1, nboot);
%                             [~, btsm1] = bootstrp(nboot, [], tt1Tar); % Bootstrap sample indices for class 1
%                             [~, btsm2] = bootstrp(nboot, [], tt2Tar); % Bootstrap sample indices for class 2
%                             labels1    = labels(1 : length(tt1Tar));
%                             labels2    = labels(length(tt1Tar)+1 : end);
%                             for k = 1:nboot % For each bootstrap sample
%                                 [~,~,~,btAUCT(k)] = perfcurve([{labels1{btsm1(:,k)}}, {labels2{btsm2(:,k)}}],...
%                                     [tt1Tar(btsm1(:,k)); tt2Tar(btsm2(:,k))], posclass);
%                                 [~,~,~,btAUCS(k)] = perfcurve([{labels1{btsm1(:,k)}}, {labels2{btsm2(:,k)}}],...
%                                     [tt1Sur(btsm1(:,k)); tt2Sur(btsm2(:,k))], posclass);
%                             end
%                             mean_btAUCTar = mean(btAUCT);
%                             mean_btAUCSur = mean(btAUCS);
%                             % Fit normal distribution to AUC values
%                             distAUCT = fitdist(btAUCT','Normal');
%                             distAUCS = fitdist(btAUCS','Normal');
%                             % Get 95% confidence intervals on bootstrap AUC value distributions
%                             AUCT_ci95 = paramci(distAUCT);
%                             ciTarUp   = AUCT_ci95(2, 1);
%                             ciTarDn   = AUCT_ci95(1, 1);
%                             AUCS_ci95 = paramci(distAUCS);
%                             ciSurUp   = AUCS_ci95(2, 1);
%                             ciSurDn   = AUCS_ci95(1, 1);
%                             
%                         case 'builtin' % Uses built-in bootstrapping in perfcurve
%                             [btAUCT, btAUCS] = deal(nan(3,1));
%                             if any(~isnan(scoreTar))
%                                 [~,~,~,btAUCT] = perfcurve(labels, scoreTar, posclass, 'NBoot', nboot);
%                                 [~,~,~,btAUCS] = perfcurve(labels, scoreSur, posclass, 'NBoot', nboot);
%                             end
%                             mean_btAUCTar  = btAUCT(1);
%                             ciTarUp        = btAUCT(3);
%                             ciTarDn        = btAUCT(2);
%                             mean_btAUCSur  = btAUCS(1);
%                             ciSurUp        = btAUCS(3);
%                             ciSurDn        = btAUCS(2);
%                             
%                     end
%                     % Save
%                     aucTar(j)     = AUCT;
%                     aucSur(j)     = AUCS;
%                     btaucTar(j)   = mean_btAUCTar;
%                     btaucSur(j)   = mean_btAUCSur;
%                     ciaucTarUp(j) = ciTarUp;
%                     ciaucTarDn(j) = ciTarDn;
%                     ciaucSurUp(j) = ciSurUp;
%                     ciaucSurDn(j) = ciSurDn;
%                     disp(['Bin ', num2str(j), ' done'])
%                 end
%                 auROCData        (1, i) = {aucTar};
%                 auROCData        (2, i) = {aucSur};
%                 btstrp_auROCData (1, i) = {btaucTar};
%                 btstrp_auROCData (2, i) = {ciaucTarUp};
%                 btstrp_auROCData (3, i) = {ciaucTarDn};
%                 btstrp_auROCData (4, i) = {btaucSur};
%                 btstrp_auROCData (5, i) = {ciaucSurUp};
%                 btstrp_auROCData (6, i) = {ciaucSurDn};
                stdROCData       (1, i) = {stdAUCT(1)};
                stdROCData       (2, i) = {stdAUCT(3)};
                stdROCData       (3, i) = {stdAUCT(2)};
                stdROCData       (4, i) = {stdAUCS(1)};
                stdROCData       (5, i) = {stdAUCS(3)};
                stdROCData       (6, i) = {stdAUCS(2)};
                
                disp(['Iteration ', num2str(i), ' done'])
            end
            
            % Set tables
            ttNames    = {'Tactile_Visual', 'Lick_NoLick', 'Touch_Light', 'TouchHit_TouchMiss',...
                'VisHit_Vis_Miss', 'TouchHit_VisHit'};
            rowNames   = {'Target', 'Surrogate'};
            rowNames_b = {'Target_mean', 'Target_CI95Up', 'Target_CI95Dn',...
                'Surrogate_mean', 'Surrogate_CI95Up', 'Surrogate_CI95Dn'}; 
            
            auROCData        = cell2table(auROCData, 'VariableNames', ttNames, 'RowNames', rowNames);
            btstrp_auROCData = cell2table(btstrp_auROCData, 'VariableNames', ttNames, 'RowNames', rowNames_b);
            stdROCData       = cell2table(stdROCData, 'VariableNames', ttNames, 'RowNames', rowNames_b);
        end
        
        
        function [auc, quickauc] = issig (class1, class2, nboot, n_comp)
            % Returns whether the two classes are significantly different
            % based on AUC analysis
            %
            % RD fall 2020
            
            if nargin < 3
                nboot = 1000;
            end
            
            % Make labels
            labels = ones(numel(class1) + numel(class2), 1);
            labels(numel(class1) + 1: end) = 2;
            
            % Set posclass
            posclass = labels(1);
%             if nanmean(class2) > nanmean(class1)
%                 posclass = labels(end);
%             end
                        
            % Get AUC
            [~,~,~,auc] = perfcurve(labels, [class1; class2], posclass, 'NBoot', nboot, 'Alpha', (0.05/n_comp)); 
            
            % Return if sig
            quickauc = auc(2) > 0.5 || auc(3) < 0.5;
            
        end
        
        
        function avgROCdata (ROCgrand, filt) % Also plots
            % To calculate and plot mean auROC traces over timebins across
            % cells and sessions
            % RD fall 2019
            
            if nargin < 2
                filt = false;
            end
            
            % Get data
            sampleLength                         = numel(table2array(ROCgrand{1}(1,1)));
            [avgT, stdT, semT, avgS, stdS, semS] = deal(nan(width(ROCgrand{1}), sampleLength)); % n pairs of trial-types
            
            % Calculate avgs and sems
            for i = 1:width(ROCgrand{1})
                thisPairT = nan(numel(ROCgrand), sampleLength);
                thisPairS = nan(numel(ROCgrand), sampleLength);
                for j = 1:numel(ROCgrand)
                    thisPairT(j, :) = table2array(ROCgrand{j}(1,i));
                    thisPairS(j, :) = table2array(ROCgrand{j}(4,i));
                end
                avgT(i, :) = nanmean(thisPairT, 1);
                stdT(i, :) = nanstd(thisPairT, 1);
                semT(i, :) = stdT(1, i) / sqrt(numel(ROCgrand));
                avgS(i, :) = nanmean(thisPairS, 1);
                stdS(i, :) = nanstd(thisPairS, 1);
                semS(i, :) = stdS(1, i) / sqrt(numel(ROCgrand));
                if filt
                    avgT(i, :) = medfilt1(avgT(i, :), 3);
                    semT(i, :) = medfilt1(semT(i, :), 3);
                    avgS(i, :) = medfilt1(avgS(i, :), 3);
                    semS(i, :) = medfilt1(semS(i, :), 3);
                end
            end
                                    
            % Plot
            time  = (linspace(-2400, 2600, size(avgT, 2))) / 1000;
            rgbcc = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560;...
                0.4660 0.6740 0.1880];
            p = cell(width(ROCgrand{1}), 1);
            figure('Renderer', 'painters', 'Position', [100 100 1200 600])
            
            % Set axis properties for both subplots
            for i = 1:2
                subplot(1,2,i)
                hold on
                plot([time(1) time(end)], [0.5 0.5], '-k', 'LineStyle', ':', 'LineWidth', 1.5)
                plot([0 0], [0 1], '-k', 'LineStyle', ':', 'LineWidth', 1.5)
                set(gca, 'xlim', [-2 2], 'ylim', [0 1], 'linewidth', 1.5, 'TickDir', 'out', 'Fontsize', 16)
                xlabel ('Time (s)', 'FontWeight', 'bold')
                ylabel ('AUC', 'FontWeight', 'bold')
                xticks ([-2 0 2])
                yticks ([0 0.5 1])
            end
            
            % Plot target whisker data
            subplot (1,2,1)
            hold on
            for i = 1:width(ROCgrand{1})
                MPlot.ErrorShade(time, avgT(i, :), semT(i, :), 'Color', rgbcc(i, :), 'Alpha', 0.3)
                p{i} = plot(time, avgT(i, :), 'Color', rgbcc(i, :), 'LineWidth', 1.75);
            end
            title 'Target whisker'
            
            % Plot surrogate whisker data
            subplot (1,2,2)
            hold on
            for i = 1:width(ROCgrand{1})
                MPlot.ErrorShade(time, avgS(i, :), semS(i, :), 'Color', rgbcc(i, :), 'Alpha', 0.3)
                p{i} = plot(time, avgS(i, :), 'Color', rgbcc(i, :), 'LineWidth', 1.75);
            end
            title 'Surrogate whisker'
                    
%             legend ([p{1} p{2} p{3} p{4} p{5}], 'Tac/Vis', 'Lick/no-Lick', 'Touch/Light', 'T. Hit/T. Miss', 'L. FA/L. CR')
%             set(legend,'Location','NorthWest')
%             legend('boxoff')
            
            subplot(1,2,1)
            text(-1.9, 0.95, 'Tac/Vis', 'Color', rgbcc(1, :), 'FontSize', 20)
            text(-1.9, 0.88, 'Lick/no-Lick', 'Color', rgbcc(2, :), 'FontSize', 20)
            text(-1.9, 0.81, 'Touch/Light', 'Color', rgbcc(3, :), 'FontSize', 20)
            text(-1.9, 0.74, 'T. Hit/T. Miss', 'Color', rgbcc(4, :), 'FontSize', 20)
            text(-1.9, 0.67,  'L. FA/L. CR', 'Color', rgbcc(5, :), 'FontSize', 20)
            
        end
        
            
        function bin = sigStart (ar, crit)
            % Returns first bin satisfying criterion
            % RD fall 2020
            
            arsum = ar;
            for i = 2:crit
                arsum(1:end-i+1) = arsum(1:end-i+1) + ar(i:end);
            end
            
            bin = find(arsum >= crit, 1);
            
            if isempty(bin)
                bin = nan;
            end
            
        end
        
        
        function onsets = getDiscOnset (btstrp_auROCData, crit)
            % Returns the bin denoting start of discriminability
            % RD fall 2020
            
            time = (linspace(-2443, 2557, 208)) / 1000;
            stim = 103;

            dat  = cell(2,6);
            
            for i=1:6
                tarErrorUp = table2array(btstrp_auROCData(2,i)) < 0.5;
                tarErrorDn = table2array(btstrp_auROCData(3,i)) > 0.5;
                surErrorUp = table2array(btstrp_auROCData(5,i)) < 0.5;
                surErrorDn = table2array(btstrp_auROCData(6,i)) > 0.5;
                try
                    tarUp      = time(dlcdata.sigStart (double(tarErrorUp(stim+1:end)), crit) + stim);
                catch
                    tarUp = nan;
                end
                try
                    tarDn      = time(dlcdata.sigStart (double(tarErrorDn(stim+1:end)), crit) + stim);
                catch
                    tarDn = nan;
                end
                dat{1, i}  = min([tarUp, tarDn]);
                try
                    surUp      = time(dlcdata.sigStart (double(surErrorUp(stim+1:end)), crit) + stim);
                catch
                    surUp = nan;
                end
                try
                    surDn      = time(dlcdata.sigStart (double(surErrorDn(stim+1:end)), crit) + stim);
                catch
                    surDn = nan;
                end
                dat{2, i}  = min([surUp, surDn]);
            end
            
            ttNames  = {'Tactile_Visual', 'Lick_NoLick', 'Touch_Light', 'TouchHit_TouchMiss',...
                'VisHit_Vis_Miss', 'TouchHit_VisHit'};
            rowNames = {'Target', 'Surrogate'};
            onsets   = cell2table(dat, 'VariableNames', ttNames, 'RowNames', rowNames);
        end
        
        
        % Plotting
        
        
        function plotWithErrors (ar, main_c, second_c)
            % RD spring 2023
           
            if nargin < 3
                second_c = 'k';
            end
            if nargin < 2
                main_c = 'k';
            end
            x_axis = linspace(1, height(ar), height(ar));
            means = mean(ar, 2, 'omitnan');
            sems = std(ar, 0, 2, 'omitnan') / sqrt(width(ar));
            plot(means, 'Color', main_c, 'Linewidth', 2);
            MPlot.ErrorShade(x_axis, means, sems, 'Alpha', 0.3, 'Color', second_c);
            
        end
                                
        
        function plotAvgs (trialAvgs, obj)
            
            tacMeans = cell2mat(trialAvgs{1,1});
            tacSEM   = cell2mat(trialAvgs{3,1}); 
            visMeans = cell2mat(trialAvgs{1,2});
            visSEM   = cell2mat(trialAvgs{3,2});
            time_tac = (cumsum(ones(length(tacMeans), 1)) * (1000/obj.userData.framerate)) - 2470;
            time_vis = (cumsum(ones(length(visMeans), 1)) * (1000/obj.userData.framerate)) - 2470;
            
            figure
            hold on
            plot (time_tac, tacMeans(:,1), 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.25);
            plot (time_vis, visMeans(:,1), 'Color', [0 0.4470 0.7410], 'LineWidth', 1.25);
            MPlot.ErrorShade(time_tac, tacMeans(:,1), tacSEM(:,1), 'Alpha', 0.5, 'Color', [0.8500 0.3250 0.0980]);
            MPlot.ErrorShade(time_vis, visMeans(:,1), visSEM(:,1), 'Alpha', 0.5, 'Color', [0 0.4470 0.7410]);
            set (gca, 'xlim', [-2500 4000], 'ylim', [9 18])
            legend ('Touch block', 'Light block')
            xlabel ('Time (ms)', 'FontWeight', 'bold')
            ylabel ('\theta^o', 'FontWeight', 'bold')
            ax = gca;
            ax.FontSize = 12;
            
            figure
            hold on
            plot (time_tac, tacMeans(:,2), 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.25);
            plot (time_vis, visMeans(:,2), 'Color', [0 0.4470 0.7410], 'LineWidth', 1.25);
            MPlot.ErrorShade(time_tac, tacMeans(:,2), tacSEM(:,2), 'Alpha', 0.5, 'Color', [0.8500 0.3250 0.0980]);
            MPlot.ErrorShade(time_vis, visMeans(:,2), visSEM(:,2), 'Alpha', 0.5, 'Color', [0 0.4470 0.7410]);
            set(gca, 'xlim', [-2500 4000], 'ylim', [8 18]) 
            legend ('Touch block', 'Light block')
            xlabel ('Time (ms)', 'FontWeight', 'bold')
            ylabel ('\theta^o', 'FontWeight', 'bold')
            ax = gca;
            ax.FontSize = 12;
            
        end
        
        
        function plotTraces (obj)
            % Plot traces of all tracked parts
            
            [dlcTraces] = obj.GetTable('DLCData');
            
            for i = 1:height(dlcTraces)
                figure
                hold on
                plot(dlcTraces.tarBase_x{i,1} , dlcTraces.tarBase_y{i,1})
                plot(dlcTraces.tarMid_x{i,1}  , dlcTraces.tarMid_y{i,1})
                plot(dlcTraces.tarEnd_x{i,1}  , dlcTraces.tarEnd_y{i,1})
                plot(dlcTraces.surMid_x{i,1}  , dlcTraces.surMid_y{i,1})
                plot(dlcTraces.surBase_x{i,1} , dlcTraces.surBase_y{i,1})
                plot(dlcTraces.nose_x{i,1}    , dlcTraces.nose_y{i,1})
                ax      = gca;
                ax.YDir = 'reverse';
            end
           
        end
        
        
        function plotGrandAvg (varargin)
            
            p = inputParser();
            p.addRequired('trialAvgs')
            p.addParameter('framerate', 500.15, @isscalar);
            p.addParameter('wind', [-2 2],  @(x) isvector(x) && isnumeric(x));
            p.addParameter('tstart', -2.5, @isscalar)
            
            p.parse(varargin{:});
            trialAvgs = p.Results.trialAvgs;
            framerate = p.Results.framerate;
            wind      = p.Results.wind;
            tstart    = p.Results.tstart;
            
            sampleInds = round((wind - tstart) * framerate);
            sampleInds = linspace(sampleInds(1), sampleInds(2), sampleInds(2) - sampleInds(1) +1);
            time       = (sampleInds / framerate) + tstart;
            sampleInds = sampleInds - 30;
                        
            titles = {'Touch block', 'Light block'};
            
            figure('Renderer', 'painters', 'Position', [100 100 1200 600])
            
            for i=1:2
                ind = i*4;
                
                cr    = cell2mat(trialAvgs{1, ind+2});
                miss  = cell2mat(trialAvgs{1, ind+1});
                fa    = cell2mat(trialAvgs{1, ind});
                hit   = cell2mat(trialAvgs{1, ind-1});
                crE   = cell2mat(trialAvgs{3, ind+2});
                missE = cell2mat(trialAvgs{3, ind+1});
                faE   = cell2mat(trialAvgs{3, ind});
                hitE  = cell2mat(trialAvgs{3, ind-1});
                
                subplot(1,2,i)
                hold on
                MPlot.ErrorShade(time, hit(sampleInds, 2), hitE(sampleInds, 2), 'Color', [0 0.4470 0.7410], 'Alpha', 0.3)
                MPlot.ErrorShade(time, fa(sampleInds, 2), faE(sampleInds, 2), 'Color', [0.4660 0.6740 0.1880], 'Alpha', 0.3)
                MPlot.ErrorShade(time, miss(sampleInds, 2), missE(sampleInds, 2), 'Color', [0.3 0.3 0.3], 'Alpha', 0.3)
                MPlot.ErrorShade(time, cr(sampleInds, 2), crE(sampleInds, 2), 'Color', [0.8500 0.3250 0.0980], 'Alpha', 0.3)
                p4 = plot(time, cr(sampleInds, 2), 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.75);
                p3 = plot(time, miss(sampleInds, 2), 'Color', [0.3 0.3 0.3], 'LineWidth', 1.75);
                p2 = plot(time, fa(sampleInds, 2), 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 1.75);
                p1 = plot(time, hit(sampleInds, 2), 'Color', [0 0.4470 0.7410], 'LineWidth', 1.75);
%                 plot([time(1) time(end)], [0 0], '-k', 'LineStyle', ':')
                plot([0 0], [0 90], '-k', 'LineStyle', ':')
                title (titles(i))
                set(gca, 'xlim', wind, 'ylim', [0 31], 'linewidth', 1.5, 'TickDir', 'out', 'Fontsize', 16)
                xlabel ('Time (s)', 'FontWeight', 'bold')
                ylabel ('\theta_{surr.} (^o)', 'FontWeight', 'bold')
                xticks ([-2 -1 0 1 2])
                yticks ([0 15 30])
                
                %                 legend ([p1 p2 p3 p4], 'Hit', 'FA', 'Miss', 'CR')
                %                 set(legend,'Location','NorthWest')
                %                 legend('boxoff')
                                
            end
            
            subplot(1,2,1)
            text(1, 29, 'Hit', 'Color', [0 0.4470 0.7410], 'FontSize', 20)
            text(1, 27, 'Miss', 'Color',[0.3 0.3 0.3], 'FontSize', 20)
            text(1, 25, 'FA', 'Color',[0.4660 0.6740 0.1880], 'FontSize', 20)
            text(1, 23, 'CR', 'Color',[0.8500 0.3250 0.0980], 'FontSize', 20)
            
        end
        
        
        function plotROCs (obj, rocData)
            
            time   = (linspace(-2443, 2557, 208)) / 1000;
            titles = {'Tactile vs. Visual', 'Lick vs. No Lick',...
                'Touch block vs. Light block', 'Tactile lick  vs. Tactile no lick',...
                'Visual lick vs. Visual Miss', 'Tactile lick vs. Visual lick'};
            
            for j = 1:2 % For target or surrogate whisker
                
                figure('Renderer', 'painters', 'Position', [100 100 1500 1000])
                for i=1:6
                    subplot(2,3,i)
                    hold on
                    
                    Target_whisker    = table2array(rocData(1,i));
                    tarErrorUp        = table2array(rocData(2,i)) - Target_whisker;
                    tarErrorDn        = Target_whisker - table2array(rocData(3,i));
                    Surrogate_whisker = table2array(rocData(4,i));
                    surErrorUp        = table2array(rocData(5,i)) - Surrogate_whisker;
                    surErrorDn        = Surrogate_whisker - table2array(rocData(6,i));
                    
                    % Convert any NaN values in error vectors to 0
                    tarErrorUp(isnan(tarErrorUp)) = 0;
                    tarErrorDn(isnan(tarErrorDn)) = 0;
                    surErrorUp(isnan(surErrorUp)) = 0;
                    surErrorDn(isnan(surErrorDn)) = 0;
                    tar2                          = Target_whisker;
                    sur2                          = Surrogate_whisker;
                    tar2(isnan(tar2))             = 0;
                    sur2(isnan(sur2))             = 0;
                    
                    % Plot
                    MPlot.ErrorShade(time, tar2, tarErrorUp, tarErrorDn, 'Color', [0 0.4470 0.7410], 'Alpha', 0.3)
                    MPlot.ErrorShade(time, sur2, surErrorUp, surErrorDn, 'Color', [0.8500 0.3250 0.0980], 'Alpha', 0.3)
                    p1 = plot(time, Target_whisker, 'Color', [0 0.4470 0.7410], 'LineWidth', 1);
                    p2 = plot(time, Surrogate_whisker, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1);
                    plot([0 0], [0 1], '-k', 'LineStyle', ':')
                    plot([time(1) time(end)], [0.5 0.5], '-k', 'LineStyle', ':')
                    
                    title (titles(i))
                    set(gca, 'xlim', [-2 2], 'ylim', [0 1])
                    xlabel ('Time (s)', 'FontWeight', 'bold')
                    ylabel ('auROC', 'FontWeight', 'bold')
                    xticks ([-2 0 2])
                    yticks ([0 0.5 1])
                    ax = gca;
                    ax.FontSize = 16;
                    legend ([p1 p2], 'Target', 'Surrogate')
                    set(legend,'Location','SouthWest')
                    legend('boxoff')
                    
                end
                
                sgtitle([obj.mouseName, '_', obj.sessionDate], 'FontSize', 14)
                
                figName = [obj.mouseName, '_', obj.sessionDate, '_auROC plots_builtin.tiff'];
                saveas (gcf, figName)
                close
            end
            
        end
        
        
        function plotrocavgs (obj, rocData, avgData)
            
            time   = (linspace(-2443, 2557, 208)) / 1000;
            titles = {'Tactile vs. Visual', 'Lick vs. No Lick',...
                'Touch block vs. Light block', 'Tactile lick  vs. Tactile no lick',...
                'Visual lick vs. Visual no lick', 'Tactile lick vs. Visual lick'};
            whiskers = {'TARGET', 'SURROGATE'};
            
            offset   = 0.4;
            
            for j = 1:2 % For target or surrogate whisker
                
                avgInd       = j + (j-1) * 5;
                aucInd       = j + (j-1) * 2;
                scale_factor = j * 10 + 2;
                
                figure('Renderer', 'painters', 'Position', [100 100 1500 1000])
                for i=1:6
                    subplot(2,3,i)
                    hold on
                    
                    whiskAvg1   = table2array(avgData(avgInd, i)) / scale_factor + offset;
                    whiskSemUp1 = (table2array(avgData(avgInd+1, i)) / scale_factor) - whiskAvg1;
                    whiskSemDn1 = whiskAvg1 - (table2array(avgData(avgInd+2, i)) / scale_factor);
                    whiskAvg2   = table2array(avgData(avgInd+3, i)) / scale_factor + offset;
                    whiskSemUp2 = (table2array(avgData(avgInd+4, i)) / scale_factor) - whiskAvg2;
                    whiskSemDn2 = whiskAvg2 - (table2array(avgData(avgInd+5, i)) / scale_factor);
                    whiskAuc    = table2array(rocData(aucInd,i));
                    whiskAucUp  = table2array(rocData(aucInd+1,i)) - whiskAuc;
                    whiskAucDn  = whiskAuc - table2array(rocData(aucInd+2,i));
                                        
                    % Convert any NaN values in error vectors to 0
                    whiskSemUp1(isnan(whiskSemUp1)) = 0;
                    whiskSemDn1(isnan(whiskSemDn1)) = 0;
                    whiskSemUp2(isnan(whiskSemUp2)) = 0;
                    whiskSemDn2(isnan(whiskSemDn2)) = 0;
                    whiskAucUp(isnan(whiskAucUp))   = 0;
                    whiskAucDn(isnan(whiskAucDn))   = 0;
                    avg1                            = whiskAvg1 + offset;
                    avg2                            = whiskAvg2 + offset;
                    auc                             = whiskAuc;
                    avg1(isnan(avg1))               = 0;
                    avg2(isnan(avg2))               = 0;
                    auc(isnan(auc))                 = 0;
                    
                    % Plot
                    MPlot.ErrorShade(time, avg1, whiskSemUp1, whiskSemDn1, 'Color',...
                        [0 0.4470 0.7410], 'Alpha', 0.3)
                    MPlot.ErrorShade(time, avg2, whiskSemUp2, whiskSemDn2, 'Color',...
                        [0.8500 0.3250 0.0980], 'Alpha', 0.3)
                    MPlot.ErrorShade(time, auc, whiskAucUp, whiskAucDn, 'Color',...
                        [0 0 0], 'Alpha', 0.3)
                    p1 = plot(time, whiskAvg1, 'Color', [0 0.4470 0.7410], 'LineWidth', 1);
                    p2 = plot(time, whiskAvg2, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1);
                    p3 = plot(time, whiskAuc, 'Color', [0 0 0], 'LineWidth', 1);
                    plot([0 0], [0 1], '-k', 'LineStyle', ':')
                    plot([time(1) time(end)], [0.5 0.5], '-k', 'LineStyle', ':')
                    
                    title (titles(i))
                    set(gca, 'xlim', [-2 2], 'ylim', [-0.2 1.5])
                    xlabel ('Time (s)', 'FontWeight', 'bold')
                    ylabel ('auROC', 'FontWeight', 'bold')
                    xticks (linspace(-2, 2, 9))
                    yticks ([0 0.5 1])
                    ax = gca;
                    ax.FontSize = 16;
                    legend ([p1 p2 p3], 'Trial-type 1', 'Trial-type 2', 'AUC')
                    set(legend,'Location','SouthWest')
                    legend('boxoff')
                    
                end
                
                %Save
                sgtitle([obj.mouseName, ' ', obj.sessionDate, ' ', whiskers{j}], 'FontSize', 30, 'FontWeight', 'bold')
                hgsave(1, [obj.mouseName, ' ', obj.sessionDate, ' ', whiskers{j}, ' auROC + avg plots.fig'], '-v7.3')
                figName = [obj.mouseName, ' ', obj.sessionDate, ' ', whiskers{j}, ' auROC + avg plots.tiff'];            
                saveas (gcf, figName)
                close
            end
            
        end
        
        
        function plotAvgSingle (avgData, mouseName, sessionName)
            % Plot avg whisker trace for a single session
            
            time   = (linspace(-2443, 2557, 208)) / 1000;
            c1 = [0.5137    0.2863    0.5608; 0.9020    0.5137    0.0706]; % Deep purple, Orange
            c2 = [0.3 0.3 0.3]; % Gray
            keys = {'Tactile', 'Visual'};
            
            for i = 4:5 % Only using TacLick/TacNoLick and VisLick/VisNoLick data
                % Get data
                whiskAvg1   = table2array(avgData(1, i));
                whiskSemUp1 = (table2array(avgData(2, i))) - whiskAvg1;
                whiskSemDn1 = whiskAvg1 - table2array(avgData(3, i));
                whiskAvg2   = table2array(avgData(4, i));
                whiskSemUp2 = table2array(avgData(5, i)) - whiskAvg2;
                whiskSemDn2 = whiskAvg2 - table2array(avgData(6, i));
                
                % Convert any NaN values in error vectors to 0
                whiskSemUp1(isnan(whiskSemUp1)) = 0;
                whiskSemDn1(isnan(whiskSemDn1)) = 0;
                whiskSemUp2(isnan(whiskSemUp2)) = 0;
                whiskSemDn2(isnan(whiskSemDn2)) = 0;
                whiskAvg1(isnan(whiskAvg1))     = 0;
                whiskAvg2(isnan(whiskAvg2))     = 0;
                
                figure('Renderer', 'painters', 'Position', [100 100 600 500])
                hold on
%                 
%                 ylimtop = max([whiskAvg1(81:144), whiskAvg2(81:144)]) + 2.5;
%                 ylimbot = min([whiskAvg1(81:144), whiskAvg2(81:144)]) - 2.5;
                % Plot
                MPlot.ErrorShade(time, whiskAvg1, whiskSemUp1, whiskSemDn1, 'Color',...
                    c1(i-3, :), 'Alpha', 0.6)
                MPlot.ErrorShade(time, whiskAvg2, whiskSemUp2, whiskSemDn2, 'Color',...
                    c2, 'Alpha', 0.6)
                p1 = plot(time, whiskAvg1, 'Color', c1(i-3, :), 'LineWidth', 2);
                p2 = plot(time, whiskAvg2, 'Color', c2, 'LineWidth', 2);
                plot([0 0.15], [8.5 8.5], '-k', 'LineWidth', 12)
%                 plot([0 0], [0 1], '-k', 'LineStyle', ':')
%                 plot([time(1) time(end)], [0.5 0.5], '-k', 'LineStyle', ':')
                
%                 set(gca, 'xlim', [-.4 1], 'ylim', [-0.2 17])
                set(gca, 'xlim', [-.4 1], 'ylim', [-8 9])
                xlabel ('Time from stim onset (s)')
                ylabel ('Target whisker angle (^o)')
                xticks (linspace(-2, 2, 9))
                yticks (linspace(-8, 8, 5))
                dlcdata.paperize()
                
                % Add legend
                legtext = [keys{i-3}, ' lick'];
%                 text(0.5, ylimbot+2, legtext, 'Fontsize', 16, 'Color', c1(i-3, :))
                text(0.5, -4, legtext, 'Fontsize', 16, 'Color', c1(i-3, :))
                legtext = [keys{i-3}, ' no lick'];
%                 text(0.5, ylimbot+1, legtext, 'Fontsize', 16, 'Color', [0.1 0.1 0.1])
                text(0.5, -5, legtext, 'Fontsize', 16, 'Color', c2)
                
                % Save
%                 cd 'E:\data\figures\main'
%                 print([keys{i-3}, ' whisker angle example'],'-depsc','-opengl')
%                 saveas(gcf,[keys{i-3}, ' whisker angle example'],'epsc')
%                 export_fig [keys{i-3}, ' whisker angle example'] '-eps'
                saveas(gcf,[mouseName, '_', sessionName, ' TARGET_baseAngles_', keys{i-3}, '_Lck-NoLck'],'epsc')
                saveas(gcf, [mouseName, '_', sessionName, ' TARGET_baseAngles_', keys{i-3}, '_Lck-NoLck.png'])
                hgsave(1, [mouseName, '_', sessionName, ' TARGET_baseAngles_', keys{i-3}, '_Lck-NoLck.fig'], '-v7.3')
                
            end
             
        end % for figure
        
        
        function plotROCSingle (save, rocData, onsets) % for figure
            
            time   = (linspace(-2443, 2557, 208)) / 1000;
            c1 = [0.5137    0.2863    0.5608; 0.9020    0.5137    0.0706]; % Deep purple, Orange
            keys = {'Tactile', 'Visual'};
            onsets = table2array(onsets (1,4:5));
            
            for i = 4:5
                % Get data
                Target_whisker    = table2array(rocData(1,i));
                tarErrorUp        = table2array(rocData(2,i)) - Target_whisker;
                tarErrorDn        = Target_whisker - table2array(rocData(3,i));
                tarErrorUp(isnan(tarErrorUp)) = 0;
                tarErrorDn(isnan(tarErrorDn)) = 0;
                tar2                          = Target_whisker;
                tar2(isnan(tar2))             = 0;
                
                figure('Renderer', 'painters', 'Position', [100 100 600 500])
                hold on
                
                % Plot
                MPlot.ErrorShade(time, tar2, tarErrorUp, tarErrorDn, 'Color', c1(i-3, :), 'Alpha', 0.6)
                p1 = plot(time, Target_whisker, 'Color', c1(i-3, :), 'LineWidth', 1.5, 'MarkerSize', 20);
                p1 = plot(time, Target_whisker, '.', 'Color', c1(i-3, :), 'MarkerSize', 10);
                plot([time(1) time(end)], [0.5 0.5], '-k', 'LineStyle', ':', 'LineWidth', 1.5)
                
                plot([onsets(i-3) onsets(i-3)], [0 1], '-k', 'LineWidth', 2.5, 'Color', c1(i-3, :))
                plot([0 0.15], [1 1], '-k', 'LineWidth', 15)

                
                set(gca, 'xlim', [-.4 1], 'ylim', [0 1])
                xlabel ('Time from stim onset (s)')
                ylabel ('Detect probability')
                xticks (linspace(-2, 2, 9))
                yticks ([0 0.5 1])
                dlcdata.paperize()
                
                % Add legend
                legtext = keys{i-3};
                text(0.6, 0.25, legtext, 'Fontsize', 16, 'Color', c1(i-3, :))
                                
                 % Save
                if save
                    cd 'E:\data\figures\main_trial'
                    %                 print([keys{i-3}, ' whisker angle example'],'-depsc','-opengl')
                    saveas(gcf, [keys{i-3}, ' auc example'],'epsc')
                    saveas(gcf, [keys{i-3}, ' auc example.tiff'])
                    %                 export_fig [keys{i-3}, ' whisker angle example'] '-eps'
                end
            end
            
        end
        
        
        function plotstds ()
            
%             rocdir = 'E:\data\matlab\dd\stdroc\';
%             rocdir = 'E:\data\matlab\test\stdroc\';
            rocdir = 'E:\data\matlab\dd\stdroc\wBonferroni\';
            files  = struct2cell(dir(rocdir));
            files  = files(1, 3:end)';
            
            [tarData, surData, tarDn, surDn, tarUp, surUp] = deal(nan(numel(files), 5));
           
                        
            for i = 1:numel(files) % For each session
                data = load ([rocdir, files{i}]);
                data = data.stdrocData;
                
                % For target whisker
                for j = 1:5 % For each trial pair
                    tarDn(i,j) = table2array(data (3, j));
                    tarUp(i,j) = table2array(data (2, j));
                    if tarDn(i,j) > 0.5
                        tarData (i, j) = table2array(data(1, j));
                    elseif tarUp(i,j) < 0.5
                        tarData (i, j) = 1 - table2array(data(1, j));
                    end
                end
                
                % For surrogate whisker
                for j = 1:5 % For each trial pair
                    surDn(i,j) = table2array(data (6, j));
                    surUp(i,j) = table2array(data (5, j));
                    if surDn(i,j) > 0.5
                        surData (i, j) = table2array(data(4, j));
                    elseif surUp(i,j) < 0.5
                        surData (i, j) = 1 - table2array(data(4, j));
                    end
                end
            end
            
%             ylabels = {'Tac/Vis','Lick/No lick','Touch/Light', 'Tactile lick/Tactile no lick', 'Visual lick/Visual no lick'};
%                         'T. Hit/V. Hit'};
            ylabels = {'Touch/Light', 'Tactile lick/Tactile no lick', 'Visual lick/Visual no lick'};
                        
            figure('Renderer', 'painters', 'Position', [100 100 1500 600])
%             subplot (2,1,1)
%             ht = heatmap(tarData', 'Colormap', jet, 'Colorlimits', [0.5 1], 'GridVisible', 'off',...
%                 'ColorbarVisible', 'off', 'MissingDataColor', [1 1 1], 'MissingDataLabel', 'not sig.',...
%                 'CellLabelColor', 'none');
% %             set(gca, 'XDisplayLabels', xticklabels, 'fontsize', 16);
% %             set(gca, 'YDisplayLabels', yticklabels, 'fontsize', 16);
%             Ax = gca;
% %             Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
%             Ax.YDisplayLabels = ylabels;
% %             xlabel 'Session'
%             Ax.FontSize = 20;
%             title 'Target whisker'
% 
%             subplot (2,1,2)
            hs = heatmap(surData(:, 3:end)', 'Colormap', parula, 'Colorlimits', [0.5 1], 'GridVisible', 'off',...
                'ColorbarVisible', 'on', 'MissingDataColor', [1 1 1], 'MissingDataLabel', 'not sig.',...
                'CellLabelColor', 'none');
%             set(gca, 'XDisplayLabels', xticklabels, 'fontsize', 16);
%             set(gca, 'YDisplayLabels', yticklabels, 'fontsize', 16);
            Ax = gca;
            Ax.YDisplayLabels = ylabels;
            Ax.FontSize = 20;
            xlabel 'Session'
            title 'Surrogate whisker'
            
%             sgtitle 'AUC for pre-stimulus theta-std'
            
            cd 'E:\data\figures\main'
            figName = 'AUC_prestim_surrogate_theta_std_4.tiff';
            saveas (gcf, figName)
            saveas(gcf,'std_heatmap_4','epsc')
            close
            
        end % for figure
        
        
        function plotOnsets ()
            
            onsdir = 'E:\data\matlab\dd\onsets\';
            files  = struct2cell(dir(onsdir));
            files  = files(1, 3:end)';
            
            combined = nan(numel(files), 2); 
            
            for i = 1:numel(files) % For each session
                data = load ([onsdir, files{i}]);
                data = data.onsets;
                
                combined(i, 1) = table2array(data(2, 4));
%                 combined(i, 2) = table2array(data(2, 4));
                combined(i, 2) = table2array(data(2, 5));
%                 combined(i, 4) = table2array(data(2, 5));
%                 combined(i, 5) = min([combined(i, 1), combined(i, 2)]);
%                 combined(i, 6) = min([combined(i, 3), combined(i, 4)]);

            end
            
            % Plot
            colors = [0.7176, 0.2745, 1.0000; 0.9294, 0.6941, 0.1255];
            figure('Renderer', 'painters', 'Position', [100 100 350 350])
            hold on
            for i = 1:2
                h = cdfplot(combined(:,i));
                set (h, 'Linewidth', 1.5, 'Color', colors(i,:))
                grid off
            end
            title ''
            set(gca, 'xlim', [0 1], 'ylim', [0 1], 'TickDir', 'out')
            xlabel ('DP Onset (s)')
            ylabel ('Cumulative fraction')
            xticks (linspace(0, 1, 6))
            yticks ([0 0.5 1.0])
            ax = gca;
            ax.FontSize = 16;
%             legend ('Tactile', 'Visual')
%             set(legend,'Location','SouthEast')
%             legend('boxoff')
            legtext = 'Tactile';
            text(0.6, 0.5, legtext, 'Fontsize', 16, 'Color', [0.7176, 0.2745, 1.0000])
            legtext = 'Visual';
            text(0.6, 0.42, legtext, 'Fontsize', 16, 'Color', [0.9294, 0.6941, 0.1255])
            
            % Save
            figName = 'DP_onsets_surrogate.tiff';
            saveas (gcf, figName)
%             
            hgsave(1, 'DP_onsets_surrogate.fig', '-v7.3') 
%             cd 'E:\data\figures\main'
%             saveas(gcf,'onsets','epsc')
            
        end % for figure
        
        
        function plotScatter (save, windDiffs, windAUC)
            
            % Get data
            if nargin < 3
                stim    = windDiffs (:, 2);
                sig     = windDiffs (:, 3);
                ylim    = [-2 2];
                ylab    = '\theta_t_a_r initial stim. difference (^o)';
                mid     = 0;
                ytick   = [-2 -1 0 1 2];
                ename   = 'windDiffs';
                tname   = '100pre_14-34post_abs-prestim.tiff';
            else
                stim = windAUC (:, 1);
                sig  = windAUC (:, 4);
                ylim = [0 1];
                ylab = 'Stimulus auROC';
                mid  = 0.5;
                ytick = [0 0.5 1];
                ename = 'windAUC';
                tname = '100pre_1-200post_abs-prestim_auc.tiff';
            end
            preStim = windDiffs (:, 1);
            mouse   = windDiffs (:, 4);
%             preSN   = abs(preStim) / max(abs(preStim));
%             stimN   = abs(stim) / max(abs(stim));
            
%             % Perform linear regression
%             rawB    = [ones(length(preStim), 1) preStim]\stim;
%             rawPred = [ones(length(preStim), 1) preStim] * rawB;
%             rawR2   = 1 - sum((stim - rawPred).^2)/sum((stim - mean(stim)).^2);
%             normB = preSN\stimN;
            
            colors = [68/255,  119/255, 170/255;...
                      102/255, 204/255, 238/255;...
                      34/255,  136/255, 51/255;...
                      204/255, 187/255, 68/255;...
                      238/255, 102/255, 119/255;...
                      170/255, 51/255,  119/255];
                  
            % Plot non-normalized data
            figure('Renderer', 'painters', 'Position', [100 100 700 700])
            hold on    
            plot([0 0], [-10 10], '-k', 'LineStyle', ':', 'Linewidth', 2)
            plot([-20 20], [mid mid], '-k', 'LineStyle', ':', 'Linewidth', 2)
            for ses = 1:numel(preStim)
                if sig(ses)
                    plot(preStim(ses), stim(ses), '.', 'color', colors(mouse(ses), :), 'Markersize', 72) 
                else
                    plot(preStim(ses), stim(ses), 'o', 'color', colors(mouse(ses), :),...
                        'Markersize', 19, 'Linewidth', 2.5)
                end
            end
            set(gca, 'xlim', [-2 6], 'ylim', ylim)
            xlabel ('\theta_t_a_r pre-stim. difference (^o)')
            ylabel (ylab)
            xticks (linspace(-2, 6, 5))
            yticks (ytick)
            dlcdata.paperize()
            %             title('Tactile no lick - Tactile lick', 'Fontsize', 20)
            if save
                cd 'E:\data\figures\main_trial'
                %                 print([keys{i-3}, ' whisker angle example'],'-depsc','-opengl')
                saveas(gcf, ename,'epsc')
                saveas(gcf, tname)
            end
            
            % Plot normalized data
%             figure('Renderer', 'painters', 'Position', [100 100 600 600])
%             hold on      
%             for ses = 1:numel(preStim)
%                 plot(preSN(ses), stimN(ses), '.', 'color', colors(mouse(ses), :), 'Markersize', 40) 
%             end
% %             plot([0 0], [-10 10], '-k', 'LineStyle', ':', 'Linewidth', 2)
% %             plot([-10 10], [0 0], '-k', 'LineStyle', ':', 'Linewidth', 2)
%             set(gca, 'xlim', [0 1], 'ylim', [0 1])
%             xlabel ('Pre-stimulus target angle difference (^o)', 'FontWeight', 'bold')
%             ylabel ('Initial stimulus target angle difference (^o)', 'FontWeight', 'bold')
%             xticks ([0 0.5 1])
%             yticks ([0 0.5 1])
%             ax = gca;
%             ax.FontSize = 16;
%             saveas(gcf, '100pre_14-34post_norm.png')
            
        end % for figure
        
        
        function plotAucvAucScatter (aucTac, aucVis)
            
            % Get data
            tac   = aucTac (:, 1);
            vis   = aucVis (:, 1);
            mouse = aucTac (:, 5);
            
            colors = [68/255,  119/255, 170/255;...
                      102/255, 204/255, 238/255;...
                      34/255,  136/255, 51/255;...
                      204/255, 187/255, 68/255;...
                      238/255, 102/255, 119/255;...
                      170/255, 51/255,  119/255];
                  
            % Plot
            figure('Renderer', 'painters', 'Position', [100 100 700 700])
            hold on    
            plot([0.5 0.5], [-10 10], '-k', 'LineStyle', ':', 'Linewidth', 2)
            plot([-20 20], [0.5 0.5], '-k', 'LineStyle', ':', 'Linewidth', 2)
            
            for ses = 1:numel(mouse)
                plot(vis(ses), tac(ses), '.', 'color', colors(mouse(ses), :), 'Markersize', 55)
            end
            
            set(gca, 'xlim', [0 1], 'ylim', [0 1], 'tickDir', 'out')
            xlabel ('Visual stimulus auROC')
            ylabel ('Tactile stimulus auROC')
            xticks (linspace(0, 1, 3))
            yticks (linspace(0, 1, 3))
            ax = gca;
            ax.FontSize = 16;
            title('Target whisker', 'Fontsize', 20)
            
            
            cd 'E:\data\figures\prestim diff vs stim diff\stimDiff vs stimDiff'
            %                 print([keys{i-3}, ' whisker angle example'],'-depsc','-opengl')
            saveas(gcf, 'Tac_v_Vis_stimDiff_Target','epsc')
            saveas(gcf, 'Tac_v_Vis_stimDiff_Target.tiff')
            hgsave(1, 'Tac_v_Vis_stimDiff_Target.fig', '-v7.3')
            
            
        end
        
        
        function plotSTDscatter (save, windDiffs, tar)
             
            % Get data
            rocdir = 'E:\data\matlab\dd\stdroc\';
            files  = struct2cell(dir(rocdir));
            files  = files(1, 3:end)';
            
            stdData       = nan(numel(files), 2);
            stdData(:, 2) = false;
            
            if tar
                row = 1;
            else
                row = 4;
            end
                                    
            for i = 1:numel(files)-1 % For each session
                data          = load ([rocdir, files{i}]);
                data          = data.stdrocData;
                stdData(i, 1) = table2array(data(row, 4));           
                tarDn         = table2array(data (row+2, 4));
                tarUp         = table2array(data (row+1, 4));
                if tarDn > 0.5 || tarUp < 0.5
                    stdData (i, 2) = true;
                end
            end
           
            
            stim = stdData (:, 1);
            sig  = stdData (:, 2);
            ylim = [0 1];
            ylab = '\theta_t_a_r pre-stim. STD auROC';
            mid  = 0.5;
            ytick = [0 0.5 1];
            ename = 'windprestimSTD';
            tname = '100pre_prestimSTD_auc.tiff';
            
            preStim = windDiffs (:, 1);
            mouse   = windDiffs (:, 4);
            colors = [68/255,  119/255, 170/255;...
                      102/255, 204/255, 238/255;...
                      34/255,  136/255, 51/255;...
                      204/255, 187/255, 68/255;...
                      238/255, 102/255, 119/255;...
                      170/255, 51/255,  119/255];
                  
            % Plot non-normalized data
            figure('Renderer', 'painters', 'Position', [100 100 700 700])
            hold on    
            plot([0 0], [-10 10], '-k', 'LineStyle', ':', 'Linewidth', 2)
            plot([-10 10], [mid mid], '-k', 'LineStyle', ':', 'Linewidth', 2)
            for ses = 1:numel(preStim)
                if sig(ses)
                    plot(preStim(ses), stim(ses), '.', 'color', colors(mouse(ses), :), 'Markersize', 72) 
                else
                    plot(preStim(ses), stim(ses), 'o', 'color', colors(mouse(ses), :),...
                        'Markersize', 19, 'Linewidth', 2.5)
                end
            end
            set(gca, 'xlim', [-2 6], 'ylim', ylim)
            xlabel ('\theta_t_a_r pre-stim. difference (^o)')
            ylabel (ylab)
            xticks ([-2 0 2 4 6])
            yticks (ytick)
            dlcdata.paperize()
            %             title('Tactile no lick - Tactile lick', 'Fontsize', 20)
            
            
            % Save
            if save
                cd 'E:\data\figures\main_trial'
                %                 print([keys{i-3}, ' whisker angle example'],'-depsc','-opengl')
                saveas(gcf, ename,'epsc')
                saveas(gcf, tname)
            end
            
        end % for figure
        
        
        function plotRT (tacRT, visRT, onsets)
            % Plots reactions times for each trial vs DP onset for that
            % session
            %
            % RD fall 2020
            
            assert (numel(unique(tacRT(:,2))) == numel(unique(visRT(:,2))), 'Number of sessions must be consistent')
            
            % Make x-axis vectors
            [tacDP, visDP] = deal([]);
            for i = 1:numel(unique(tacRT(:,2)))
                tacDP = [tacDP; repmat(onsets(i, 1), sum(tacRT(:, 2) == i), 1)];
                visDP = [visDP; repmat(onsets(i, 2), sum(visRT(:, 2) == i), 1)];
            end
            
            % Get mouseID vector
            mouseIDs = dlcdata.makeMouseIDvector();
            
            % Find median reaction times
            tacMed = dlcdata.getMedianRTs(tacRT) / 1000;
            visMed = dlcdata.getMedianRTs(visRT) / 1000;
            
            % Define colors
            c      = [0.4745    0.0275    0.6392; 0.8510    0.3255    0.0980];
            colors = [68/255,  119/255, 170/255;...
                      102/255, 204/255, 238/255;...
                      34/255,  136/255, 51/255;...
                      204/255, 187/255, 68/255;...
                      238/255, 102/255, 119/255;...
                      170/255, 51/255,  119/255];
            
            % Plot
            titles = {'Tactile', 'Visual'};
            
            figure('Renderer', 'painters', 'Position', [100 100 1000 500])
            for i=1:2
                subplot (1,2,i)
                hold on
                if i==1
                    plot (tacDP, tacRT(:, 1) / 1000, '.', 'MarkerSize', 10, 'Color', c(1, :))
                    for ses = 1:numel(mouseIDs)
                        plot(onsets(ses, 1), tacMed(ses), '*', 'color',...
                             colors(mouseIDs(ses), :), 'Markersize', 12, 'LineWidth', 3)
                    end
                else
                    plot (visDP, visRT(:, 1) / 1000, '.', 'MarkerSize', 10, 'Color', c(2, :))
                    for ses = 1:numel(mouseIDs)
                        plot(onsets(ses, 2), visMed(ses), '*', 'color',...
                             colors(mouseIDs(ses), :), 'Markersize', 12, 'LineWidth', 3)
                    end
                end
                set(gca, 'xlim', [0 0.5], 'ylim', [0 1.5])
                xlabel ('DP onset (s)')
                ylabel ('Reaction time (s)')
                xticks (linspace(0, 0.5, 6))
                yticks (linspace(0, 1.5, 6))
                dlcdata.paperize()
                title(titles{i})
            end
            
            % Save
            figName = 'Reaction_times vs surrogateDP onsets_wMedians.png';
            saveas (gcf, figName)
            saveas(gcf,'Reaction_times vs surrogateDP onsets_wMedians','epsc')
            hgsave(1, 'Reaction_times vs surrogateDP onsets_wMedians.fig', '-v7.3')
            
        end % for figure
               
         
        % misc
        
                
        function DD = createSuperObj (save, mainPath)
            % Collates multiple dlcdata objects into 
            % one giant MSessionExplorer object
            %
            % RD fall 2019
                        
            if nargin == 0
                save = false;
            elseif nargin < 2
                mainPath = 'E:\high_speed_video\';
            end
            
            sesData = dlcdata.readSesData ();
            
            for i = 1:height(sesData)
                if ~isempty(cell2mat(sesData{i, 1}))
                    thisMouse   = cell2mat(sesData{i, 1});
                    thisSession = num2str(sesData{i, 2});
                else
                    continue
                end
                
                sessionName = [thisMouse, '_', thisSession];
                ddfile      = [mainPath, sessionName, '\', sessionName, '_dlcdata.mat'];
                
                try
                    load(ddfile);
                catch
                    error ('Could not load dlcdata object')
                end
                
                if i == 1
                    ddArray = dd;
                    continue
                end
                
                ddArray = [ddArray; dd];
            end
            
            DD = Merge (ddArray);            
            
            if save
                save([mainPath, 'se_allSessions.mat'], 'DD', '-v7.3', '-nocompression');
            end
            
        end
        
        
        function sampleVids (sesData, maindir, vidsperses)
            % Displays a list of randomly selected files from
            % subdirectories in the directory provided
            % RD fall 2019
            
            if nargin < 3
                vidsperses = 5;
            end
            
            if nargin < 1 || isempty(maindir)
                maindir    = 'D:\high_speed_video\';
            end
            
            for ses = 1:height(sesData)
                videodir     = ([maindir, '\_videos\avi files\', cell2mat(table2array(sesData(ses, 1))), '_',...
                    num2str(table2array(sesData(ses, 2))), '\']);
                vidInds      = nan(1, vidsperses);
                sesdir       = dir(videodir);
                numvids      = length(sesdir) - 2;
                vidInds(:,:) = round(1+(numvids-1).*rand(1, vidsperses));
                for j = 1:vidsperses
                    vid = [sesdir(3).folder, '\', sesdir(vidInds(1,j)+2).name];
                    vid = strrep(vid, '\', '/');
                    disp (['''', vid, ''','])
                end
                
            end
        end
        
        
        function getFilenames (ttArray, folder, n)
            
            tinds = find(ttArray);
            if n > 0
                n = min(n, numel(tinds));
            end
            
            csv = dir(folder);
            csv = csv(4:end);
            csv = csv(tinds);
            
            used = [];
            for i = 1:n
                this = round(rand(1) * length(csv));
                if ~any(find(this==used))
                    disp([num2str(this), '. ', csv(this).name])
                    used(end+1) = this;
                end
            end
            disp(['Found ', num2str(numel(used)), ' trials'])
            
        end
        
        
        function filled = fillNans (data, analyzeWind)
            
            filled = data;
            for j = 1:size (data,1)
                if length(data{j,1}) < analyzeWind
                    nanTemp = nan(1, analyzeWind - length(data{j,1}));
                    filled{j,1}(end+1:analyzeWind) = nanTemp;
                    filled{j,2}(end+1:analyzeWind) = nanTemp;
                end
            end
            
        end
        
        
        function [mean, semUp, semDn] = meansem (ar)
            % Returns mean and upper and lower bounds of sem
            % RD fall 2020
            
            mean  = nanmean(ar);
            std   = nanstd(ar);
            sem   = std / sqrt(numel(ar));
            semUp = mean + sem;
            semDn = mean - sem;
        end
        
        
        function [TacLick, TacNoLick, VisLick, VisNoLick] = quickTTMap (obj)
            
            trialTypes = {'TTT', 'TTV', 'TTN', 'VTT', 'VTV', 'VTN', 'TVT', 'TVV', 'TVN', 'VVT', 'VVV', 'VVN'};
            dat        = array2table(obj.GetColumn('behavValue', trialTypes), 'VariableNames', trialTypes);
            TacLick    = logical(dat.TTT + dat.TTV + dat.VTT + dat.VTV);
            TacNoLick  = logical(dat.TTN + dat.VTN);
            % data = data(ttData.Tac, :);
            VisLick    = logical(dat.TVT + dat.TVV + dat.VVT + dat.VVV);
            VisNoLick  = logical(dat.TVN + dat.VVN);

        end
        
        
        function mouseIDs = makeMouseIDvector(mainPath)
            
            if nargin < 1
                mainPath  = 'E:\data\matlab\dd\';
            end
            sesData = dlcdata.readSesData(mainPath);
            
            mouseIDs = nan(height(sesData), 1);
            mice     = unique(sesData{:, 1});
            nmouse   = numel(mice);
            for mouse = 1:nmouse
                mouseIDs(cell2mat(cellfun(@(x) strcmp(x, mice{mouse}), sesData{:, 1}, 'Uni', 0)), 1) = mouse;
            end
            
        end
        
        
        function paperize ()
            
            set(gca, 'TickDir', 'out', 'LineWidth', 1.2, 'TickLength', [0.02, 0.01])
            ax = gca;
            ax.FontSize = 20;
            
        end
        
      
    end
    
end