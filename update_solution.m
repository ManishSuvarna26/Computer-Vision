function [Pnew,Unew] = update_solution(deltav,P,U)
% Computes updated cameras Pnew and 3D points Unew, from the old solution
% P,U and the parameter increment deltav. If cameras are not calibrated, K
% will still be constant throughout the update step.

% Extract the incremental 3D point update parameters from first part of deltav
% These are the first (3*size(U,2)-1) elements of deltav, but yield the
% total 3*size(U,2) point parameter increments, when combined with a fix 0
% increment for the first parameter.
% delta_pointvar = [0; deltav(1:(3*size(U,2)-1))]; % shape: (3*n_pts, 1)
% delta_pointvar = reshape(delta_pointvar, size(U(1:3,:))); % shape: (3, n_pts)

% Extract the incremental 3D point update parameters (a,b,c,t1,t2,t3) for
% each camera from the last part of deltav. 6 parameters for each camera,
% and the first camera is fix (0 increments).
% delta_camvar = [0;0;0;0;0;0;deltav(3*size(U,2):end)]; % shape: (6*n_cams, 1)
delta_camvar = deltav;
delta_camvar = reshape(delta_camvar,[6 length(P)]); % shape: (6, n_cams)

% U = pflat(U); % Make sure that last coordinate is 1.
% Update 3D points with the increments.
% Unew = pextend(U(1:3,:) + delta_pointvar);

% Tangent basis for the rotation manifold.
Ba = [0 1 0; -1 0 0; 0 0 0];
Bb = [0 0 1; 0 0 0; -1 0 0];
Bc = [0 0 0; 0 0 1; 0 -1 0];

Pnew = cell(size(P));
for i=1:length(P);
    % Determine camera calibration. If camera is not calibrated, K will
    % still be constant throughout the update step.
    [K,Ri] = rq(P{i});
    R0 = Ri(:,1:3);
    t0 = Ri(:,4);

%     Extract individual camera parameters for current camera:
    delta_a = delta_camvar(1,i);
    delta_b = delta_camvar(2,i);
    delta_c = delta_camvar(3,i);
    delta_t = delta_camvar(4:6,i);
    
    % The manifold of rotation matrices is linearized around the previous
    % rotation estimate R0, and the updated rotation R is given by multiplying
    % R0 with the exponential matrix of the skew-symmetric matrix
    % delta_a*Ba + delta_b*Ba + delta_c*Bc.
    R = expm(delta_a*Ba + delta_b*Ba + delta_c*Bc)*R0;

    % The updated camera translation is simply the t-part incremented.
    t = t0 + delta_t;

    % Bring back the (unchanged) calibration matrix.
    Pnew{i} = K*[R t];
end
