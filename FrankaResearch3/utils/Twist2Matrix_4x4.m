function OUTPUT = Twist2Matrix_4x4(TWIST)
    %TWIST2MATRIX_4X4 此处显示有关此函数的摘要
    %   此处显示详细说明
    OUTPUT=[Vector2Matrix_3x3(TWIST(1:3,1)),TWIST(4:end,1);0,0,0,1];
end

