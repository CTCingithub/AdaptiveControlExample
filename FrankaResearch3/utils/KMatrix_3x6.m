function Output = KMatrix_3x6(Omega)
    Output=[Omega(1),Omega(2),Omega(3),0,0,0;
        0,Omega(1),0,Omega(2),Omega(3),0;
        0,0,Omega(1),0,Omega(2),Omega(3)];
end