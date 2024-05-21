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
o2 = 0.25; fe2 = 0.02; Sr = 5.13; no3 = 0.3;
y0 = [o2, fe2, Sr, no3];
nSpecies = 4;

% time span
tspan = [0 3600];

% ODEs
[t, y] = ode45(@(t, y) reaction_system(t, y, k1, k3, k5, k9, k10, kdo, kea, kn, kI), tspan, y0);

% Key parts
num_fast_species = 1;
contributions = zeros(length(t),  nSpecies);

a = zeros(nSpecies, nSpecies, length(t));
b = zeros(nSpecies, nSpecies, length(t));

for i = 1:length(t)
    J_numeric = evaluate_jacobian(y(i, :), k1, k3, k5, k9, k10, kdo, kea, kn, kI);
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
plot(t, y(:,4), 'b', 'DisplayName', 'no3');
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
plot(t, contributions(:, 4), 'b', 'DisplayName', 'no3');
xlabel('Time');
ylabel('Contribution to Fast Modes');
legend show;
title('Contributions of Species to Fast Modes');
hold off;

% ODEs
function dydt = reaction_system(~, y, k1, k3, k5, k9, k10, kdo, kea, kn, kI)
    o2 = y(1);
    fe2 = y(2);
    Sr = y(3);
    no3 = y(4);
    
    do2_dt = -k1*o2/(o2+kdo) - k3*o2*fe2 - k5*o2*Sr;
    dfe2_dt = -4*k3*o2*fe2 - 5*k10*no3/(no3+kn)*fe2/(fe2+Sr+kea)*kI/(o2+kI);
    dSr_dt = -4*k5*o2*Sr - 5*k10*no3/(no3+kn)*Sr/(fe2+Sr+kea)*kI/(o2+kI);
    dno3_dt = -k9*no3/(no3+kn)*kI/(o2+kI) - k10*no3/(no3+kn)*(fe2+Sr)/(fe2+Sr+kea)*kI/(o2+kI);
    
    dydt = [do2_dt; dfe2_dt; dSr_dt; dno3_dt];
end

% Jacobian
function J = evaluate_jacobian(y, k1, k3, k5, k9, k10, kdo, kea, kn, kI)
    o2 = y(1);
    fe2 = y(2);
    Sr = y(3);
    no3 = y(4);
    
    J = zeros(4, 4);

    J(1,1) = -k1*kdo/(o2+kdo)^2 - k3*fe2 - k5*Sr;
    J(2,1) = -k3*o2;
    J(3,1) = -k5*o2;
    J(4,1) = 0;

    J(1,2) = -4*k3*fe2 + 5*k10*no3/(no3+kn)*fe2/(fe2+Sr+kea)*kI/(o2+kI)^2;
    J(2,2) = -4*k3*o2 - 5*k10*no3/(no3+kn)*kI/(o2+kI)/(fe2+Sr+kea)^2;
    J(3,2) = -5*k10*no3/(no3+kn)*kI/(o2+kI)*(fe2+Sr)/(fe2+Sr+kea)^2;
    J(4,2) = -5*k10*kI/(o2+kI)*fe2/(fe2+Sr+kea)*no3/(no3+kn)^2;

    J(1,3) = -4*k5*o2 + 5*k10*no3/(no3+kn)*Sr/(fe2+Sr+kea)*kI/(o2+kI)^2;
    J(2,3) = -5*k10*no3/(no3+kn)*kI/(o2+kI)*(fe2+Sr)/(fe2+Sr+kea)^2;
    J(3,3) = -4*k5*o2 - 5*k10*no3/(no3+kn)*kI/(o2+kI)*(fe2+Sr)/(fe2+Sr+kea)^2;
    J(4,3) = -5*k10*kI/(o2+kI)*Sr/(fe2+Sr+kea)*no3/(no3+kn)^2;

    
    J(1,4) = -k9*kI/(o2+kI)*kn/(no3+kn)^2 - k10*(fe2+Sr)/(fe2+Sr+kea)*kI/(o2+kI)*kn/(no3+kn)^2;
    J(2,4) = -k10*no3/(no3+kn)*kI/(o2+kI)*kea/(fe2+Sr+kea)^2;
    J(3,4) = -k10*no3/(no3+kn)*kI/(o2+kI)*kea/(fe2+Sr+kea)^2;
    J(4,4) = -k9*kI/(o2+kI)*kn/(no3+kn)^2 - k10*(fe2+Sr)/(fe2+Sr+kea)*kI/(o2+kI)*kn/(no3+kn)^2;
end