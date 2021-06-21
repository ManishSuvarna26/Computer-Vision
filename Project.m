% Project

% Note: If the code gives any error, please run it again. Thank You.
%% RANSAC + DLT
clc;
clear all;
n_mat = 9; % number of given images
for i = 1:n_mat
    datai = sprintf('data%d.mat', i);
    imgi = sprintf('img%d.jpg',i);
    data{i} = load(datai);
    img{i} = imread(imgi);
end
n_object = 7;
n_iterations = 300;
threshold = 5e-3;
best_count = 0;
LMiterations = 25;
lambda = 10e-15;

for i = 1:n_object
    for j = 1:n_mat
        x = data{j}.u{i}; % Goes to an image and takes the image points of one object
        n_points = length(x);
        x = [x; ones(1,n_points)];
        X = data{j}.U{i};% Goes to an image and takes the 3d points of one object
        m = size(X,2);
        X_homo = [X; ones(1,m)];
        Xmean = mean(X,2);
        Xtilde =  X - repmat(Xmean, [1 m]);
        s = 3;
        % RANSAC
        best_count = 0;
        for k = 1:n_iterations
            randind = [];
            for k = 1:s
                randind = [randind fix(rand*m + 1)]; 
            end
            xs = x(:,randind);
            Xs = Xtilde(1:3,randind);
            % Minimal solver gives the calibrated solutions [R t]
            min_sol = minimalCameraPose(xs,Xs);
            n_sol = length(min_sol);
            n_inliers = [];
            % Checking which solution has more inliers
            for l = 1:n_sol
                P{l} = min_sol{l};
                [K{l}, R{l}] = rq(P{l});
                
                x_proj{l} = pflat(P{l}*X_homo);
                err{l} = (sum(x_proj{l} - x).^2).^0.5; 
                inliers{l} = err{l} <= threshold;
                n_inliers = [n_inliers sum(inliers{l})];
                [most_inliers, maxind] = max(n_inliers);
            end
            % Save the best solution
            if most_inliers > best_count
                best_count = most_inliers;
                largest_X = X_homo(:,inliers{maxind});
                largest_x = x(:,inliers{maxind}); % gathering the outlier free correspondances
                P_ransac =P{maxind};
            end
        end
        % Minimal solver solution on inliers only
        P_est{i,j} = P_ransac;
        Xinliers{i,j} = largest_X;
        xinliers{i,j} = largest_x;
 
    %DLT on the outlier free data
    n1 = length(xinliers{i,j});
    M = zeros(3*n1, (3*size(Xinliers{i,j},1) + n1));
    
    for r = 1:length(Xinliers{i,j})
        M(3*r,9:12) = Xinliers{i,j}(:,r)';
        M(3*r-1,5:8) = Xinliers{i,j}(:,r)';
        M(3*r-2,1:4) = Xinliers{i,j}(:,r)';
        M(3*r-2:3*r,12+r) = -xinliers{i,j}(:,r);
    end
    [U,S,V] = svd(M);
    DLT_sol = V(:,end);
  
    P_normalized = reshape(DLT_sol(1:12),[4 3])';
    R_dlt = P_normalized(:,1:3);
   % Setting contraints on rotation matrix
    %SVD on rotation matrix to have det=1.
    [Up, Sp, Vp] = svd(R_dlt);
    
    Sp = eye(3);
    R_dlt = Up*Sp*Vp';
    t = P_normalized(:,end)/P_normalized(end,end);
    P_est_DLT{i,j} = [R_dlt t];
    P_est_DLT{i,j} = P_est_DLT{i, j}*sign(det(P_est_DLT{i, j}(:,1:3)));
    

    end
  
end

% Ground Truth & Bounding boxes
for i = 1:n_object
    for j = 1:n_mat
        P_gts{i,j} = data{j}.poses{i};
        bounding_boxes{i,j} = data{j}.bounding_boxes{i};
    end
end

%% LM on RANSAC + Minimal Solver solution

P_est_P3P_LM = cell(size(P_est));

for i = 1:n_object
    for j = 1:n_mat
        for k = 1:LMiterations
            P = {P_est{i,j}};
            U = Xinliers{i,j};
            u = {xinliers{i,j}};
            [err_LM(k), res] = ComputeReprojectionError(P,U,u);
            [r,J] = LinearizeReprojErr(P,U,u);
            C = J'*J + lambda*speye(size(J,2));
            c=J'*r;
            deltav = -C\c;
            [Pnew{i,j}] = update_solution(deltav,P,U);
            P_est_P3P_LM{i,j} = cell2mat(Pnew{i,j});
        end
        disp({i,j});
    end
end


%% LM on DLT solution

P_est_DLT_LM = cell(size(P_est_DLT));

for i = 1:n_object
    for j = 1:n_mat
        for k = 1:LMiterations
            P = {P_est_DLT{i,j}};
            U = Xinliers{i,j};
            u = {xinliers{i,j}};
            [err_LM(k), res] = ComputeReprojectionError(P,U,u);
            [r,J] = LinearizeReprojErr(P,U,u);
            C = J'*J + lambda*speye(size(J,2));
            c=J'*r;
            deltav = -C\c;
            [Pnew{i,j}] = update_solution(deltav,P,U);
            P_est_DLT_LM{i,j} = cell2mat(Pnew{i,j});
        end
        disp({i,j});
    end
end


        
%% Evaluation & Table of Scores
scores_DLT = cell(1,9);
scores_P3P = cell(1,9);
scores_DLT_LM = cell(1,9);
scores_P3P_LM = cell(1,9);
for i=1:9
    disp("Image "+ i + ":");
    scores_DLT{i} = eval_pose_estimates(P_gts(:,i), P_est_DLT(:,i),bounding_boxes(:,i));
    scores_P3P{i} = eval_pose_estimates(P_gts(:,i), P_est(:,i),bounding_boxes(:,i));
    scores_DLT_LM{i} = eval_pose_estimates(P_gts(:,i), P_est_DLT_LM(:,i),bounding_boxes(:,i));
    scores_P3P_LM{i} = eval_pose_estimates(P_gts(:,i), P_est_P3P_LM(:,i),bounding_boxes(:,i));
end
DLT_tab = zeros(n_object,n_mat);
P3P_tab = zeros(n_object,n_mat);
DLT_LM_tab = zeros(n_object,n_mat);
P3P_LM_tab = zeros(n_object,n_mat);
for i = 1:n_mat
    DLT_tab(:,i) = cell2mat(scores_DLT{i})';
    P3P_tab(:,i) = cell2mat(scores_P3P{i})';
    DLT_LM_tab(:,i) = cell2mat(scores_DLT_LM{i})';
    P3P_LM_tab(:,i) = cell2mat(scores_P3P_LM{i})';
end

P3P_total_avg = sum(P3P_tab(:))/numel(P3P_tab);
DLT_total_avg = sum(DLT_tab(:))/numel(DLT_tab);
DLT_LM_total_avg = sum(DLT_LM_tab(:))/numel(DLT_LM_tab);
P3P_LM_total_avg = sum(P3P_LM_tab(:))/numel(P3P_LM_tab);
P3P_avg = mean(P3P_tab,1);
DLT_avg = mean(DLT_tab,1);
DLT_LM_avg = mean(DLT_LM_tab,1);
P3P_LM_avg = mean(P3P_LM_tab,1);
P3P_tab = round([P3P_tab; P3P_avg],1);
DLT_tab = round([DLT_tab; DLT_avg],1);
DLT_LM_tab = round([DLT_LM_tab; DLT_LM_avg],1);
P3P_LM_tab = round([P3P_LM_tab; P3P_LM_avg],2);
T_P3P = array2table(P3P_tab,'VariableNames',{'Img1','Img2','Img3','Img4','Img5','Img6','Img7','Img8','Img9'},...
    'RowNames',{'Ape';'Can';'Cat';'Duck';'Eggbox';'Glue';'Holepuncher';'Average'});

T_DLT = array2table(DLT_tab,'VariableNames',{'Img1','Img2','Img3','Img4','Img5','Img6','Img7','Img8','Img9'},...
    'RowNames',{'Ape';'Can';'Cat';'Duck';'Eggbox';'Glue';'Holepuncher';'Average'});

T_DLT_LM = array2table(DLT_LM_tab,'VariableNames',{'Img1','Img2','Img3','Img4','Img5','Img6','Img7','Img8','Img9'},...
    'RowNames',{'Ape';'Can';'Cat';'Duck';'Eggbox';'Glue';'Holepuncher';'Average'});

T_P3P_LM = array2table(P3P_LM_tab,'VariableNames',{'Img1','Img2','Img3','Img4','Img5','Img6','Img7','Img8','Img9'},...
    'RowNames',{'Ape';'Can';'Cat';'Duck';'Eggbox';'Glue';'Holepuncher';'Average'});

T_P3P

T_P3P_LM

T_DLT

T_DLT_LM
%% Bounding Boxes for RANSAC + Minimal Solver

for i = 1:n_mat
     draw_bounding_boxes(img{i},P_gts(:,i),P_est(:,i),bounding_boxes(:,i));
end
%% Bounding Boxes for RANSAC + Minimal Solver + LM

for i = 1:n_mat
     draw_bounding_boxes(img{i},P_gts(:,i),P_est_P3P_LM(:,i),bounding_boxes(:,i));
end
%% Bounding Boxes for DLT + Enforcing Rotation matrix constraints

for i = 1:n_mat
     draw_bounding_boxes(img{i},P_gts(:,i),P_est_DLT(:,i),bounding_boxes(:,i));
end

%% Bounding Boxes for DLT + Enforcing Rotation matrix constraints + LM

for i = 1:n_mat
     draw_bounding_boxes(img{i},P_gts(:,i),P_est_DLT_LM(:,i),bounding_boxes(:,i));
end
%% Histogram for each method

figure(1);
subplot(2,2,1);
histogram(P3P_tab,50);
xlabel('Evaluation Score');
ylabel('Number of Pose estimation');
title('RANSAC + Minimal Solver');
subplot(2,2,2);
histogram(DLT_tab,100);
xlabel('Evaluation Score');
ylabel('Number of Pose estimation');
title('DLT + Enforcing Rotation Constraints');
subplot(2,2,3);
histogram(P3P_LM_tab,50);
xlabel('Evaluation Score');
ylabel('Number of Pose estimation');
title('RANSAC + Minimal Solver + LM');
subplot(2,2,4);
histogram(DLT_LM_tab,100);
xlabel('Evaluation Score');
ylabel('Number of Pose estimation');
title('DLT + Enforcing Rotation Constraints + LM');


