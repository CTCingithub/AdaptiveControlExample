function DynRegVec = DynRegVec(theta_1, theta_2, v_1, v_2, a_1, a_2)
    DynRegVec = [
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        a_1;
        0;
        9.8*cos(theta_1);
        0;
        -9.8*sin(theta_1);
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        0;
        a_1 + a_2;
        a_1 + a_2;
        3.4*a_1.*cos(theta_2) + 1.7*a_2.*cos(theta_2) - 3.4*v_1.*v_2.*sin(theta_2) - 1.7*v_2.^2.*sin(theta_2) + 9.8*cos(theta_1 + theta_2);
        1.7*a_1.*cos(theta_2) + 1.7*v_1.^2.*sin(theta_2) + 9.8*cos(theta_1 + theta_2);
        -3.4*a_1.*sin(theta_2) - 1.7*a_2.*sin(theta_2) - 3.4*v_1.*v_2.*cos(theta_2) - 1.7*v_2.^2.*cos(theta_2) - 9.8*sin(theta_1 + theta_2);
        -1.7*a_1.*sin(theta_2) + 1.7*v_1.^2.*cos(theta_2) - 9.8*sin(theta_1 + theta_2);
        0;
        0;
        2.89*a_1 + 16.66*cos(theta_1);
        0
        ];
end
