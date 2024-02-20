clc
clear all 
close all

function [transition_matrix, state_space] = tauchen(n, mu, rho, sigma)
     % tauchenの手法で関数を離散化
     % n:点の個数
     % mu:AR(1)の平均
     % rho:AR(1)の係数
     % sigma:誤差項の標準偏差

    m =1 / sqrt(1 - rho^2);
     % 状態空間の作成
    state_space = linspace(mu - m*sigma, mu + m*sigma, n)';
     
     %グリッドの距離
    d =(state_space(n) - state_space(1))/ (n - 1);
     % 遷移確率行列の計算
    transition_matrix = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if j == 1
                transition_matrix(i, j) = normcdf((state_space(1) - rho*state_space(i) + d/2) / sigma);
            elseif j == n
                transition_matrix(i, j) = 1 - normcdf((state_space(n) - rho*state_space(i) - d/2) / sigma);
            else
                z_low = (state_space(j) - rho*state_space(i) - d/2) / sigma;
                z_high = (state_space(j) - rho*state_space(i) + d/2) / sigma;
                transition_matrix(i, j) = normcdf(z_high) - normcdf(z_low);
            end
        end
    end
end


function param = setPar(sigma, beta, delta, alpha, rho, a_l, a_u, NH, NA)
    % 各パラメータの値
 if nargin < 9
    sigma = 1.50;   % リスク回避度
    beta = 0.98;    % 時間的割引率
    delta = 0.03;   % 減耗率
    alpha = 0.25;   % 資本分配率
    rho = 0.6;      % 能力値係数
    a_l = 0;        % 貯蓄下限
    a_u = 20;       % 貯蓄上限
    NH = 2;         % 能力値グリッド数
    NA = 401;       % 貯蓄グリッド数
 end
    % 能力値の標準偏差
    sigma_eps = sqrt(0.6*(1-rho^2));

    % tauchenからlog hのグリッドおよび遷移確率を計算
    [pi, h] = tauchen(NH, -0.7, rho, sigma_eps);
    h = exp(h); % log hからhへ
    % hの推移から定常分布
    probst = ones(NH,1) / NH; % 今期確率
    test = 10.0; % 分布収束の初期基準
    while test > 1e-8 % 分布収束の最終基準
        probst_new = zeros(NH,1); % 来期確率
        for ih = 1:NH % 今期
            for ihp = 1:NH % 来期
                probst_new(ihp) = probst_new(ihp) + pi(ih, ihp)*probst(ih);
                
            end
        end

        test = max(abs(probst_new - probst)); % 分布の差異
        probst = probst_new; % 分布を更新
    end

    HH = sum(h.*probst); % 人的資源量

    param.sigma = sigma; param.beta = beta; param.delta = delta;
    param.alpha = alpha; param.probst = probst;
    param.a_l = a_l; param.a_u = a_u; param.NH = NH; param.NA = NA;
    param.pi = pi; param.h = h; param.HH = HH;
    
end