function DynRegVec = DynRegVec(States)
    StateCell = num2cell(States);
    [theta_1, theta_2, v_1, v_2, phi_1, phi_2, psi_1, psi_2] = deal(StateCell{:});
    DynRegVec = [
        psi_1;
        9.8*cos(theta_1);
        -9.8*sin(theta_1);
        psi_1 + psi_2;
        -1.7*phi_2.*v_2.*sin(theta_2) + 3.4*psi_1.*cos(theta_2) + 1.7*psi_2.*cos(theta_2) - 1.7*(phi_1.*v_2 + phi_2.*v_1).*sin(theta_2) + 9.8*cos(theta_1 + theta_2);
        -1.7*phi_2.*v_2.*cos(theta_2) - 3.4*psi_1.*sin(theta_2) - 1.7*psi_2.*sin(theta_2) - 1.7*(phi_1.*v_2 + phi_2.*v_1).*cos(theta_2) - 9.8*sin(theta_1 + theta_2);
        2.89*psi_1 + 16.66*cos(theta_1);
        0;
        0;
        0;
        psi_1 + psi_2;
        1.7*phi_1.*v_1.*sin(theta_2) + 1.7*psi_1.*cos(theta_2) + 9.8*cos(theta_1 + theta_2);
        1.7*phi_1.*v_1.*cos(theta_2) - 1.7*psi_1.*sin(theta_2) - 9.8*sin(theta_1 + theta_2);
        0
        ];
end
