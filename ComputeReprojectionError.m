function [err,res] = ComputeReprojectionError(P,U,u)
%Compute the reprojection error from the current solution P,U and the
%imagedata u. The value of each squared residual is in res.

% Inputs:
% P: cell array of (3,4) matrices         - Camera matrices for the different views
% U: (4, n_pts)                           - Projective 3D points (homogeneous coordinates)
% u: cell array of (2, n_pts) mattrices   - Euclidean 2D points (homogeneous coordinates also work if final element is 1)

% Outputs:
% err: Sum of squared errors for all residuals (all points in all views)
% res: (1, n_views*n_pts) - Squared residuals (all points in all views)

err = 0;
res = [];
for i = 1:length(P);
    uu = u{i};
    vis = isfinite(uu(1,:));
    err = err + ...
        sum(((P{i}(1,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(1,vis)).^2) + ...
        sum(((P{i}(2,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(2,vis)).^2)    ;
    res = [res ((P{i}(1,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(1,vis)).^2 + ...
            ((P{i}(2,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(2,vis)).^2];
end
