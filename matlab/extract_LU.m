function [L, U] = extract_LU(A_lu)
% EXTRACT_LU Extracts L and U from in-place LU matrix (no pivoting)
%   [L, U] = extract_LU(A_lu)
%   L: lower triangular with ones on the diagonal
%   U: upper triangular (including diagonal)

    n = size(A_lu,1);
    L = tril(A_lu, -1) + eye(n);
    U = triu(A_lu);
end

% Example usage:
% A_lu = [2 -1 3; 2 2 -2.5; -1 0.5 3.5];
% [L, U] = extract_LU(A_lu);
% disp('L =');
% disp(L);
% disp('U =');
% disp(U);
