function Output = Vtilde_10x1(rVec,gVec,TranformationMat)
    Output=transpose([zeros(1,6),...
        -transpose(gVec)*TranformationMat(1:3,1:3),...
        -transpose(gVec)*rVec]);
end