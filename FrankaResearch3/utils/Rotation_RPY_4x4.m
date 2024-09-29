function OUTPUT = Rotation_RPY_4x4(RPY)
    %ROTATION_RPY_4X4 此处显示有关此函数的摘要
    %   此处显示详细说明
    OUTPUT= MatrixExp_4x4([1;0;0;0;0;0],RPY(1,1))*...
        MatrixExp_4x4([0;1;0;0;0;0],RPY(2,1))*...
        MatrixExp_4x4([0;0;1;0;0;0],RPY(3,1));
end

