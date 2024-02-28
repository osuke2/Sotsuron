clc
clear all 
close all

param = setPar(); 
[decisions, r, w, KK, HH, mu] = model_solution(param);

h = param.h;
delta = param.delta;
a_l = param.a_l; a_u = param.a_u; NA = param.NA; NH = param.NH;
a = linspace(a_l, a_u, NA)';


% 各グリッドでの収入を計算
income = [(r*a + w*h(1)); (r*a + w*h(2))];
income = income'; % 転置

% まとめる
zip_lists = sortrows([income(:), mu(:)]);
% ほどく
pinc = zip_lists(:, 1);
pmu = zip_lists(:, 2);


figure;
plot(pinc, pmu);
title('INCOME DISTRIBUTION');
xlabel('INCOME LEVEL');
ylabel('% OF AGENTS');

% GDPを初期化
GDP = 0;


% 各貯蓄グリッドと各能力グリッドにおいて、所得と確率密度関数を乗じて総和を取る
for ia = 1:NA
    for ih = 1:NH
        % 消費
        consume = (1 + r) * a(ia) + w * h(ih);
        % 総消費量
        GDP = GDP + consume * mu(ia, ih);
    end
end

% 減耗量を加える
GDP = GDP + delta * KK;
disp(['GDP: ', num2str(GDP)]);

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
function [aplus,iaplus,c]=solve_household_gs(param,r,w)


    % パラメータ格納
    a_l = param.a_l; a_u = param.a_u; pi = param.pi; delta = param.delta;
    beta = param.beta; sigma = param.sigma; h = param.h; NA = param.NA;
    NH = param.NH;

    % 貯蓄グリッド
    a = linspace(a_l, a_u, NA)';

    %　初期効用関数
    util = -10000.0 * ones(NA, NA, NH);

        % 今期と来期の貯蓄とhから効用を計算
    for ia = 1:NA
        for iap = 1:NA
            for ih = 1:NH
                cons = w * h(ih) + (1.0 + r) * a(ia) - a(iap);
                if cons > 0 % 消費は正
                    util(iap, ia, ih) = cons^ (1.0 - sigma) / (1.0 - sigma);
                end
            end
        end
    end


    % いくつか初期化
    v = zeros(NA, NH);
    aplus = zeros(NA, NH);
    c = zeros(NA, NH);
    v_new = zeros(NA, NH);
    iaplus_new = -10000 * ones(NA, NH);
    iaplus = -10000 * ones(NA, NH);
    reward = zeros(NA, NA, NH);

    % 政策関数探し
    test = 10;
    while test ~= 0 % 収束するまで
        for ia = 1:NA
            for ih = 1:NH
                reward(:, ia, ih) = util(:, ia, ih);
                for ihp = 1:NH
                    reward(:, ia, ih) = reward(:, ia, ih)+beta*pi(ih,ihp)*v(:,ihp);
                end
            end
        end
        % 効用最大化
        for ia = 1:NA
            for ih = 1:NH
                [v_new(ia, ih), iaplus_new(ia, ih)] = max(reward(:, ia, ih)); % 最大値とそれを与える値
            end
        end
        % 最適貯蓄量
        test = max(max(abs(iaplus_new - iaplus))); 
        v = v_new;
        iaplus = iaplus_new;
    end

    aplus = a(iaplus);

    % 最適消費量
    for ia = 1:NA
        for ih = 1:NH
            c(ia, ih) = w*h(ih) + (1.0 + r) * a(ia) - aplus(ia, ih);
        end
    end
end

function mu = get_distribution(param, decisions)
    a_l = param.a_l;
    a_u = param.a_u;
    pi = param.pi;
    NA = param.NA;
    NH = param.NH;
    iaplus = decisions{2};

    a = linspace(a_l, a_u, NA);

    test = 10; % 初期test値
    mu = ones(NA, NH) / NA / NH; % 初期分布

    % 定常分布を求める
    while test > 1e-8
        mu_new = zeros(NA, NH); 
        % a,h,h'についても繰り返す
        for ia = 1:NA
            for ih = 1:NH
                for ihp = 1:NH
                    % 来期の分布をアップデート
                    mu_new(iaplus(ia, ih), ihp) = mu_new(iaplus(ia, ih), ihp) + pi(ih, ihp) * mu(ia, ih);
                end
            end
        end
        test = max(max(abs(mu_new - mu))); % 分布の差異を求める
        mu = mu_new; 
    end
end

function [decisions, r, w, KK, HH, mu] = model_solution(param)
    alpha = param.alpha;
    delta = param.delta;
    HH = param.HH; 
    phi = 0.2; 
    toler = 1e-3; 
    test = 10; 
    KK = 10.0; 
    fprintf('ITERATING ON KK\n\n');
    fprintf('  metric    Kold      Knew\n');
    fprintf('--------------------------------\n');

    while test > toler
        w = (1 - alpha) * KK^alpha * HH^(-alpha);
        r = alpha * KK^(alpha - 1) * HH^(1 - alpha) - delta;

        %家計の問題
        [aplus, iaplus, c] = solve_household_gs(param, r, w);
        
        %定常分布
        mu = get_distribution(param, {[], iaplus});

        % 資本量を更新
        KK_new = sum(sum(mu.*aplus));

        % 資本量を求める
        test = abs((KK_new - KK) / KK); % 資本量と更新した資本量の差
        fprintf('  %.4f   %.4f   %.4f\n', test, KK, KK_new);
        KK = phi * KK_new + (1 - phi) * KK; 
    end

    decisions = {aplus, iaplus, c};
end