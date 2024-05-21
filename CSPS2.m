% Units: s,L,mmol
k1 = 1.39e-6; 
k2 = 8.3e-6;
k3 = 1.24e-3;
k4 = 3.4;
k5 = 0.2115e-3;
k6 = 0.05823;
k71 = 10;
k72 = 1.5e-3;
k8 = 1.75e-6;
k9 = 7.8e-10;
k10 = 3.5e-7;
kdo = 0.165;
kea = 0.11;
kn = 0.01;
kI = 1.6e-5;

% Initial concentrations
o2 = 0.25; fe2 = 0.02; Sr = 5.13; cr6 = 0.038; acr6 = 0; cr3 = 0;
y0 = [o2, fe2, Sr, cr6, acr6, cr3];
nSpecies = 6;

% time span
tspan = [0 3600];

% ODEs
[t, y] = ode45(@(t, y) reaction_system(t, y, k1, k3, k5, k71, k72, k8, kdo), tspan, y0);

% Key parts
num_fast_species = 1;
contributions = zeros(length(t), nSpecies);

a = zeros(nSpecies, nSpecies, length(t));
b = zeros(nSpecies, nSpecies, length(t));

for i = 1:length(t)
    J_numeric = evaluate_jacobian(y(i, :), k1, k3, k5, k71, k72, k8, kdo);
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
plot(t, y(:,2), 'g', 'DisplayName', 'fe2');
plot(t, y(:,3), 'k', 'DisplayName', 'Sr');
plot(t, y(:,4), 'c', 'DisplayName', 'cr6');
plot(t, y(:,5), 'y', 'DisplayName', 'acr6');
plot(t, y(:,6), 'm', 'DisplayName', 'cr3');
xlabel('Time');
ylabel('Concentration');
legend show;
title('Concentration of each Species');
hold off;

figure;
plot(t, contributions(:, 1), 'r', 'DisplayName', 'o2');
hold on;
plot(t, contributions(:, 2), 'g', 'DisplayName', 'fe2');
plot(t, contributions(:, 3), 'k', 'DisplayName', 'Sr');
plot(t, contributions(:, 4), 'c', 'DisplayName', 'cr6');
plot(t, contributions(:, 5), 'y', 'DisplayName', 'acr6');
plot(t, contributions(:, 6), 'm', 'DisplayName', 'cr3');
xlabel('Time');
ylabel('Contribution to Fast Modes');
legend show;
title('Contributions of Species to Fast Modes');
hold off;

% ODEs
function dydt = reaction_system(~, y, k1, k3, k5, k71, k72, k8, kdo)
    o2 = y(1);
    fe2 = y(2);
    Sr = y(3);
    cr6 = y(4);
    acr6 = y(5);
    cr3 = y(6);
    
    do2_dt = -k1*o2/(o2+kdo) - k3*o2*fe2 - k5*o2*Sr;
    dfe2_dt = -4*k3*o2*fe2;
    dSr_dt = -4*k5*o2*Sr - 3*k8*Sr*acr6;
    dcr6_dt = -k71*cr6 + k72*acr6;
    dacr6_dt = k71*cr6 - k72*acr6 - k8*Sr*acr6;
    dcr3_dt = k8*Sr*acr6;
    
    dydt = [do2_dt; dfe2_dt; dSr_dt; dcr6_dt; dacr6_dt; dcr3_dt;];
end

% Jacobian
function J = evaluate_jacobian(y, k1, k3, k5, k71, k72, k8, kdo)
    o2 = y(1);
    fe2 = y(2);
    Sr = y(3);
    cr6 = y(4);
    acr6 = y(5);
    cr3 = y(6);
   
   J = [-k1*kdo/(o2+kdo)^2-k3*fe2-k5*Sr,    -4*k3*fe2,  -4*k5*Sr,              0,     0,              0;
        -k3*o2,                             -4*k3*o2,   0,                     0,     0,              0;
        -k5*o2,                             0,          -4*k5*o2 - 3*k8*acr6,  0,     -k8*acr6,       k8*acr6;
        0,                                  0,          0,                     -k71,  k71,            0;
        0,                                  0,          -3*k8*Sr,              k72,   -k72 - k8*Sr,   0;
        0,                                  0,          0,                     0,     0,              0];
end