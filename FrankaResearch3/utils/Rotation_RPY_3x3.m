function OUTPUT = Rotation_RPY_3x3(RPY)
    %ROTATION_RPY_3X3 此处显示有关此函数的摘要
    %   此处显示详细说明
    OUTPUT=MatrixExp_3x3([1;0;0],RPY(1,1))*...
        MatrixExp_3x3([0;1;0],RPY(2,1))*...
        MatrixExp_3x3([0;0;1],RPY(3,1));
end

