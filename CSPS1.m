% Units: s,L,mmol
k1 = 1.39e-6; 
kdo = 0.165;
k2 = 8.3e-6;
k3 = 1.24e-3;
k4 = 3.4;
k5 = 0.2115e-3;
k6 = 0.05823;

% Initial concentrations
o2 = 0.25; h2o2 = 0; fe2 = 0.02; Sr = 5.13;
y0 = [o2, h2o2, fe2, Sr];
nSpecies = 4;

% time span
tspan = [0 3600];

% ODEs
[t, y] = ode45(@(t, y) reaction_system(t, y, k1, kdo, k2, k3, k4, k5, k6), tspan, y0);

% Key parts
num_fast_species = 1;
contributions = zeros(length(t), length(y0));

a = zeros(nSpecies, nSpecies, length(t));
b = zeros(nSpecies, nSpecies, length(t));

for i = 1:length(t)
    J_numeric = evaluate_jacobian(y(i, :), k1, kdo, k2, k3, k4, k5, k6);
    [V, D] = eig(J_numeric);
    eigenvalues = diag(D);
    [~, idx] = sort(abs(eigenvalues), 'descend');
    V_fast = V(:, idx(1:num_fast_species));
    a(:,:,i) = V;
    b(:,:,i) = inv(V)';
    A_f = a(:, 1:num_fast_species, i);
    B_f = b(1:num_fast_species, :, i);
    for j = 1:size(V_fast, 1)
        contributions(i, j) = max(abs(V_fast(j, :)));
    end
end

% Plot
figure;
plot(t, y(:,1), 'r', 'DisplayName', 'o2');
hold on;
plot(t, y(:,2), 'b', 'DisplayName', 'h2o2');
plot(t, y(:,3), 'g', 'DisplayName', 'fe2');
plot(t, y(:,4), 'k', 'DisplayName', 'Sr');
xlabel('Time');
ylabel('Concentration');
legend show;
title('Concentration of each Species');
hold off;

figure;
plot(t, contributions(:, 1), 'r', 'DisplayName', 'o2');
hold on;
plot(t, contributions(:, 2), 'b', 'DisplayName', 'h2o2');
plot(t, contributions(:, 3), 'g', 'DisplayName', 'fe2');
plot(t, contributions(:, 4), 'k', 'DisplayName', 'Sr');
xlabel('Time');
ylabel('Contribution to Fast Modes');
legend show;
title('Contributions of Species to Fast Modes');
hold off;

% ODEs
function dydt = reaction_system(~, y, k1, kdo, k2, k3, k4, k5, k6)
    o2 = y(1);
    h2o2 = y(2);
    fe2 = y(3);
    Sr = y(4);
    
    do2_dt = -k1*o2/(o2+kdo)-k3*o2*fe2-k5*o2*Sr;
    dh2o2_dt = (k1/5000)*o2/(o2+kdo)-k2*h2o2+k3*o2*fe2-k4*h2o2*fe2+k5*o2*Sr-k6*h2o2*Sr;
    dfe2_dt = -2*k3*o2*fe2-2*k4*h2o2*fe2;
    dSr_dt = -2*k5*o2*Sr-2*k6*h2o2*Sr;
    
    dydt = [do2_dt; dh2o2_dt; dfe2_dt; dSr_dt];
end

% Transpose Jacobian matrix
function J = evaluate_jacobian(y, k1, kdo, k2, k3, k4, k5, k6)
    o2 = y(1);
    h2o2 = y(2);
    fe2 = y(3);
    Sr = y(4);
    
    J = [-k1*kdo/(o2+kdo)^2-k3*fe2-k5*Sr, (k1/5000)*kdo/(o2+kdo)^2, -k3*o2, -k5*o2;
         k1/(5000*(o2+kdo)^2), -k2-k4*fe2-k6*Sr, -k4*h2o2, -k6*h2o2;
         -k3*fe2, k3*o2, -2*k3*o2-2*k4*h2o2, 0;
         -k5*Sr, k5*o2, 0, -2*k5*o2-2*k6*h2o2];
end
