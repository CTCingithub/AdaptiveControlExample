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
    g.*cos(theta_1);
    0;
    -g.*sin(theta_1);
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
    2*a_1.*l1.*cos(theta_2) + a_2.*l1.*cos(theta_2) + g.*cos(theta_1 + theta_2) - 2*l1.*v_1.*v_2.*sin(theta_2) - l1.*v_2.^2.*sin(theta_2);
    a_1.*l1.*cos(theta_2) + g.*cos(theta_1 + theta_2) + l1.*v_1.^2.*sin(theta_2);
    -2*a_1.*l1.*sin(theta_2) - a_2.*l1.*sin(theta_2) - g.*sin(theta_1 + theta_2) - 2*l1.*v_1.*v_2.*cos(theta_2) - l1.*v_2.^2.*cos(theta_2);
    -a_1.*l1.*sin(theta_2) - g.*sin(theta_1 + theta_2) + l1.*v_1.^2.*cos(theta_2);
    0;
    0;
    l1.*(a_1.*l1 + g.*cos(theta_1));
    0
    ]
end
