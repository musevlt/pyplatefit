SUBROUTINE average(a, b, out)
    implicit none
    integer a
    real*8 b, out

    out=(a + b)/2

    write(*,*) a, b
    write(*,*) out

    END SUBROUTINE