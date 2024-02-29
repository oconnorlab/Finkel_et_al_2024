%% Figure S3 in Finkel et. al.

% Transition analysis
% After eight trials, a drop of water from the correct response port was released in the 9th trials.
% These block cue trials were no included in the following analysis.

% load data array
load('E:\CM_NeuralActivity_Analysis\Data_array\AllTrials_25bin_noSmooth\data_array.mat')
% Import setting                
S = 2; % number of stimuli: tactile and visual stimuli
D = 3; % number of decisions: lick (including right and left lick) and no lick
C = 2; % number of contexts: respond-to-touch and respond-to-light blocks

recSite = 'left S1';

u = data_all(strcmp(data_all.recSite, recSite), :);
maxTrialNum = max(cell2mat(u{:,5}),[],'all');
firingRates = [];
numOfTrials = [];
trialNum_tBlock = [];
trialNum_vBlock = [];
transition = [];
for session=1:size(u,1)
    firingRates_session = u{session,4}{1,1};
    numOfTrials_session = u{session,5}{1,1};
    N = size(firingRates_session,1); % number of neurons
    trialNum_tBlock_session = u{session,6}{1,1};
    trialNum_vBlock_session = u{session,7}{1,1};        
    transition_session = u{session,8}{1,1};
    for n = 1:N
        for s = 1:S
            for d = 1:D
                for c = 1:C
                    firingRates_session(n,s,d,c,:,numOfTrials_session(n,s,d,c)+1:maxTrialNum) = nan;
                    trialNum_tBlock_session(n,s,d,c,numOfTrials_session(n,s,d,c)+1:maxTrialNum) = nan;
                    trialNum_vBlock_session(n,s,d,c,numOfTrials_session(n,s,d,c)+1:maxTrialNum) = nan;
                    transition_session(n,s,d,c,numOfTrials_session(n,s,d,c)+1:maxTrialNum) = nan;
                end
            end
        end
    end         
    firingRates = [firingRates;firingRates_session];
    numOfTrials = [numOfTrials;numOfTrials_session];
    trialNum_tBlock = [trialNum_tBlock;trialNum_tBlock_session];
    trialNum_vBlock = [trialNum_vBlock;trialNum_vBlock_session];
    transition = [transition; transition_session];
end

% firingRatesAverage = nanmean(firingRates, 6);

%% histogram
tts = {[1 1 1] [1 3 1] [1 1 2] [1 2 2] [1 3 2]}; % TTT (tHit), TTN (tMiss), VTT (tFA, rule error), VTV (tFA, compulsive), VTN (tCR)
tt_names = {'tHit' 'tMiss' 'tFA (rule)' 'tFA (compulsive)' 'tCR'};
u = data_all(strcmp(data_all.recSite, recSite), :);
figure('Position',[0 0 600 200*length(tts)])
for tt = 1:length(tts)
    trialNum = [];
    if tts{tt}(3) == 1 % tBlock
        for session=1:size(u,1)
            trialNum_session = u{session,6}{1,1};
            [~,~,trialNum_new] = find(trialNum_session(1,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
            trialNum = [trialNum;trialNum_new];
        end   
          
    else % vBlock
        for session=1:size(u,1)
            trialNum_session = u{session,7}{1,1};
            [~,~,trialNum_new] = find(trialNum_session(1,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
            trialNum = [trialNum;trialNum_new];
        end   
    end
    subplot(length(tts),2,tt*2-1)
    mask = trialNum <9;
    h_tt_early = histogram(trialNum(mask),[1:2:70],'FaceColor','r'); hold on;
    h_tt_late = histogram(trialNum(~mask),[1:2:70],'FaceColor','b'); hold off;
    max_tt = max([h_tt_early.Values h_tt_late.Values]);
    min_tt = min([h_tt_early.Values h_tt_late.Values]);
%     ylim([0 round(max_tt+ 0.1*(max_tt - min_tt))]);
    ylim([0 30]);
    ylabel({tt_names{tt}; 'Counts'})
    xticks([1 9 20:10:70])
    set(gca, 'box','off', 'TickDir','out')
    if tt == length(tts)
        xlabel('Number of trial after block switch')
    end
end        
%% grand average 
% zScore:(x- mean(baseline))/std(baseline) 
% bootstrapping

bin = 0.025;
timeWindow = [-1 2.5];
time = timeWindow(1)+bin:bin:timeWindow(2);
time_bw = [time, fliplr(time)];
baselineWindow = [-1 0];
baseline_startBin = (baselineWindow(1)+1)*(1/bin) + 1;
baseline_endBin = (baselineWindow(2)+1)*(1/bin);
stimWindow = [0 0.15];

N = size(firingRates,1); % number of neurons
Baseline_recSite= firingRates(:,:,:,:,baseline_startBin:baseline_endBin,:);
Baseline_mean = nanmean(Baseline_recSite(:,:),2);
Baseline_std = nanstd(Baseline_recSite(:,:),0,2);

% early, late groups
colors_group = {'r' 'b'};
rng(17); % control random number generation
nboot = 1000; % nboot
for tt = 1:length(tts)
    firingRates_tt = squeeze(firingRates(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),:,:));
    K = size(firingRates_tt,3);
    subplot(length(tts),2,tt*2)
    if tts{tt}(3) == 1 % tBlock
        trialNum_tt = squeeze(trialNum_tBlock(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
    else  % vBlock
        trialNum_tt = squeeze(trialNum_vBlock(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
    end
    for g=1:2
        if g==1 % early group
            isGroup = trialNum_tt <9;
        else % late group
            isGroup = trialNum_tt >=9;
        end
        firingRates_group = firingRates_tt;
        for n=1:N
            isGroup_unit = isGroup(n,:);
            for k=1:K
                isGroup_trial = isGroup_unit(k);
                if isGroup_trial == 0
                    firingRates_group(n,:,k)=nan;
                end
            end
        end
        X = nanmean(firingRates_group,3); % average across trials
        X_normalized = (X - Baseline_mean)./Baseline_std; 
        % remove units that do not have the "g" group of the "tt" trial type (NaN)
        isNan = isnan(X_normalized);
        X_normalized = reshape(X_normalized(~isNan),[],size(X_normalized,2));
        % bootstrapping for each time bin 
        X_normalized = num2cell(X_normalized,1); 
        bootstat = cellfun(@(x) bootstrp(nboot,@mean,x), X_normalized, 'UniformOutput', false);      
        bootstat_sorted = cellfun(@(x) sort(x), bootstat, 'UniformOutput', false);
        bootstrap_summary = cellfun(@(x) [mean(x), x(nboot*0.025), x(nboot*0.975)]', bootstat_sorted,'UniformOutput', false);
        bootstrap_summary = cell2mat(bootstrap_summary); % rows: time bin, columns: mean, lower CI, upper CI           
        
        plot(time,bootstrap_summary(1,:),colors_group{g}); hold on;
        fill(time_bw, [bootstrap_summary(2,:), fliplr(bootstrap_summary(3,:))],...
            colors_group{g},'FaceAlpha', 0.3, 'Linestyle', 'none');
        plot(stimWindow, [2 2],'k','Linewidth',2)
        set(gca, 'box','off', 'TickDir','out')
        xlim([-0.25 1]);ylim([-0.5 2]);
        xticks(-0.25:0.25:1)
        xticklabels({[] 0 [] 0.5 [] 1})
        yticks(-0.5:0.5:2)
        ylabel('Mean Z-score')
        if tt == length(tts)
            xlabel('Time from stimulus onset (s)')
        end
    end
end
% sgtitle('S1') 

%% Save figure  
mainDir = 'C:\Users\Yiting\YitingData\Finkel_revision';
FigPath = fullfile(mainDir,'Figure_transition_25bin_cue_9th_v2');
print(FigPath,'-dpdf','-painters','-loose');
 

%% Permutation test 
% O'Connor et al., Neuron, 2010 (Fig. S7D)
% Determine whether the mean PSTHs for early and lare trials for each neuron were siginificant, as measured by the Euclidean
% distance between them (Foffani and Moxon, 2004; Sandler, 2008)
% clearvars -except firingRates numOfTrials trialNum_tBlock trialNum_vBlock transition
bin = 0.025; % sec
timeWindow = [-1 2.5];
analysisWindow = [0 0.5];
binStart =(analysisWindow(1)-timeWindow(1))/bin+1;
binEnd = (analysisWindow(2)-timeWindow(1))/bin;
N = size(firingRates,1); % number of neurons

tts = {[1 1 1] [1 3 1] [1 1 2] [1 2 2] [1 3 2]}; % TTT (tHit), TTN (tMiss), VTT (tFA, rule error), VTV (tFA, compulsive), VTN (tCR)
tt_names = {'tHit' 'tMiss' 'tFA (rule)' 'tFA (compulsive)' 'tCR'};

rng(17); % control random number generation
nShuffle = 1000; 

for tt = 1:length(tts)
    firingRates_tt = squeeze(firingRates(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),binStart:binEnd,:));
    numOfTrials_tt = squeeze(numOfTrials(:,tts{tt}(1),tts{tt}(2),tts{tt}(3)));
    
    if tts{tt}(3) == 1 % tBlock
        trialNum_tt = squeeze(trialNum_tBlock(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
    else  % vBlock
        trialNum_tt = squeeze(trialNum_vBlock(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
    end
    
    for n=1:N
        numOfTrials_tt_neuron = numOfTrials_tt(n);
        trialNum_tt_neuron = trialNum_tt(n,1:numOfTrials_tt_neuron);
        isEarly = trialNum_tt_neuron <9;
        isLate = trialNum_tt_neuron >=9;
        firingRates_tt_neuron = squeeze(firingRates_tt(n,:,1:numOfTrials_tt_neuron));
        FR_early = mean(firingRates_tt_neuron(:,isEarly),2);
        FR_late = mean(firingRates_tt_neuron(:,isLate),2);
        distance = norm(FR_early - FR_late);
        
        for shuffle=1:nShuffle
            trialNum_shuffled = trialNum_tt_neuron(randperm(numOfTrials_tt_neuron));
            isEarly_shuffled = trialNum_shuffled <9;
            isLate_shuffled = trialNum_shuffled >=9;
            FR_early_shuffled = mean(firingRates_tt_neuron(:,isEarly_shuffled),2);
            FR_late_shuffled = mean(firingRates_tt_neuron(:,isLate_shuffled),2);
            distance_shuffled(shuffle) = norm(FR_early_shuffled - FR_late_shuffled);
        end
        
%         distnace_shuffled_sorted = sort(distance_shuffled);
%         distnace_shuffled_CI_upper = distnace_shuffled_sorted(:,round(0.975*nShuffle));
%         distnace_shuffled_CI_lower = distnace_shuffled_sorted(:,round(0.025*nShuffle));
        if ~isnan(distance)
%             isSig(n,tt) = double((distance - distnace_shuffled_CI_upper)*(distance -  distnace_shuffled_CI_lower)>0);  
            pValues(n,tt) = sum(distance_shuffled>=distance)/length(distance_shuffled);% one tailed 
        else
%             isSig(n,tt) = nan;
            pValues(n,tt) = nan;
        end
        
        distances(n,tt) = distance;
        distances_shuffled(n,tt) = {distance_shuffled};
    end    
end

%% Get how many neurons with significant difference between the early and late PSTHs
alpha = 0.05;
for tt = 1:length(tts)
    N = size(pValues,1);
    alpha_adjusted = alpha/N; % Bonferroni correction
    isSig = pValues(:,tt) < alpha_adjusted;
    
    summary(tt,1) = {tt_names{tt}}; % trial type
    summary(tt,2) = {sum(isSig)}; % number of signigicant neurons
    summary(tt,3) = {sum(isSig)/N}; % percentage of significant neurons
    
end
%% Cumulative histograms of the p-value for the null hypothesis (H0) of no difference (“no discrim”) between the PSTHs
alpha = 0.05;

figure('Position',[0 0 300 200*length(tts)])
for tt = 1:length(tts)
    subplot(length(tts),1,tt)
    [f,x] = ecdf(pValues(:,tt));
    ecdf(pValues(:,tt))
    N = size(pValues,1);
    ind = length(x(x<alpha/N)); % Bonferroni correction
    if ind~=0
        text(0.1,0.5,num2str(round(f(ind),2)))
    else
        text(0.1,0.5,'0')
    end
    ylabel({'Cum fraction'; 'of neurons'},'FontSize',12);
    xlim([0 1]);
    ylim([0 1]);
    xline(alpha/N,'--k'); 
    set(gca,'box','off','TickDir','out', 'XTick',[.05 0.5 1], 'YTick',[0:0.2:1])
    title([])
    if tt==length(tts)
        xlabel('P-value','FontSize',12);
    else 
        xlabel([])
    end
    title(tt_names{tt})
end
%% Save figure  
mainDir = 'C:\Users\Yiting\YitingData\Finkel_revision';
FigPath = fullfile(mainDir,'Figure_transition_pValues_0-p5_Bonferroni_cue_9th');
print(FigPath,'-dpdf','-painters','-loose');

%% Plot the distribution of the Euclidean distances under the null hypothesis and the real Euclidean distance for each unit
mainDir = 'C:\Users\Yiting\YitingData\Finkel_revision\distance_distribution_unit';
for n=1:N
    figure('Position',[0 0 300 200*length(tts)])
    for tt = 1:length(tts)
        if ~isnan(distances(n,tt))
            subplot(length(tts),1,tt)
            h = histogram(distances_shuffled{n,tt},'normalization','probability'); hold on;
            xline(distances(n,tt),'LineWidth',1);
            ylabel({tt_names{tt} 'Probability'})
            x = max(h.Data)- 0.2*max(h.Data);
            y = max(h.Values) - 0.2*max(h.Values);
            text(x,y,num2str(pValues(n,tt)))
        end
        if tt==length(tts)
            xlabel('Euclidean distance');
        else 
            xlabel([])
        end
        if tt==1
            title(['unit\_' num2str(n)])
        end
    end
    sgtitle(['unit\_' num2str(n)])
    FigPath = fullfile(mainDir,['unit_' num2str(n) '.pdf']);
    print(FigPath,'-dpdf','-painters','-loose');
end
%%
% % 50 ms bin
% % 10 ms bin and 50 ms bin generate similar results
% 
% bin = 0.01; % sec
% timeWindow = [-1 2.5];
% analysisWindow = [0 0.5];
% binStart =(analysisWindow(1)-timeWindow(1))/bin+1;
% binEnd = (analysisWindow(2)-timeWindow(1))/bin;
% N = size(firingRates,1); % number of neurons
% 
% tts = {[1 1 1] [1 3 1] [1 1 2] [1 2 2] [1 3 2]}; % TTT (tHit), TTN (tMiss), VTT (tFA, rule error), VTV (tFA, compulsive), VTN (tCR)
% tt_names = {'tHit' 'tMiss' 'tFA (rule)' 'tFA (compulsive)' 'tCR'};
% 
% rng(17); % control random number generation
% nShuffle = 1000; 
% 
% for tt = 1:length(tts)
%     firingRates_tt = squeeze(firingRates(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),binStart:binEnd,:));
%     numOfTrials_tt = squeeze(numOfTrials(:,tts{tt}(1),tts{tt}(2),tts{tt}(3)));
%     
%     if tts{tt}(3) == 1 % tBlock
%         trialNum_tt = squeeze(trialNum_tBlock(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
%     else  % vBlock
%         trialNum_tt = squeeze(trialNum_vBlock(:,tts{tt}(1),tts{tt}(2),tts{tt}(3),:));
%     end
%     
%     for n=1:N
%         numOfTrials_tt_neuron = numOfTrials_tt(n);
%         trialNum_tt_neuron = trialNum_tt(n,1:numOfTrials_tt_neuron);
%         isEarly = trialNum_tt_neuron <=10;
%         isLate = trialNum_tt_neuron >10;
%         firingRates_tt_neuron = squeeze(firingRates_tt(n,:,1:numOfTrials_tt_neuron));
%         for k=1:numOfTrials_tt_neuron
%             for bins=1:10
%                 firingRates_tt_neuron_new(bins,k) = mean(firingRates_tt_neuron((bins-1)*5+1:5*bins,k));
%             end
%         end
%         FR_early = mean(firingRates_tt_neuron_new(:,isEarly),2);
%         FR_late = mean(firingRates_tt_neuron_new(:,isLate),2);
%         distance = norm(FR_early - FR_late);
%         
%         for shuffle=1:nShuffle
%             trialNum_shuffled = trialNum_tt_neuron(randperm(numOfTrials_tt_neuron));
%             isEarly_shuffled = trialNum_shuffled <=10;
%             isLate_shuffled = trialNum_shuffled >10;
%             FR_early_shuffled = mean(firingRates_tt_neuron_new(:,isEarly_shuffled),2);
%             FR_late_shuffled = mean(firingRates_tt_neuron_new(:,isLate_shuffled),2);
%             distance_shuffled(shuffle) = norm(FR_early_shuffled - FR_late_shuffled);
%         end
%         
%         distnace_shuffled_sorted = sort(distance_shuffled);
%         distnace_shuffled_CI_upper = distnace_shuffled_sorted(:,round(0.975*nShuffle));
%         distnace_shuffled_CI_lower = distnace_shuffled_sorted(:,round(0.025*nShuffle));
%         isSig(n,tt) = (distance -  distnace_shuffled_CI_upper)*(distance -  distnace_shuffled_CI_lower)>0;        
%     end    
% end
% 
% sum(isSig,1)
