function x = back_substitution(U, b)
% BACK_SUBSTITUTION Solves Ux = b using back substitution
%   U: upper triangular matrix (n x n)
%   b: right-hand side vector (n x 1)
%   x: solution vector (n x 1)

    n = length(b);
    x = zeros(n, 1);

    % Check if U is square
    [m, p] = size(U);
    if m ~= p
        error('Matrix U must be square');
    end

    % Back substitution algorithm
    for i = n:-1:1
        if U(i,i) == 0
            error('Zero diagonal element encountered!');
        end
        x(i) = (b(i) - U(i,i+1:end) * x(i+1:end)) / U(i,i);
    end
end

% Example usage
% Uncomment below to test the function:
% U = [2 -1 3; 0 1 4; 0 0 -2];
% b = [5; 6; -4];
% x = back_substitution(U, b)
