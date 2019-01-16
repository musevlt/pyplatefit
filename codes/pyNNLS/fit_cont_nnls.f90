 real(kind=8) function do_continuum_fit(ebv)
!
!     This is the one-dimensional function called by brent.
!
      implicit none
      real(kind=8) :: ebv
      integer :: iflag

!     Since the 1D minimization routine won't know about the
!     NNLS minimization, I need to pass all relevant parameters for
!     the NNLS call below in a common block

      integer :: m,n,mda,nb
      real(kind=8) :: lambda(20000), flux(20000)
      real(kind=8) :: modellib(20000, 100), x(20000), zz(200000), w(20000)
      integer :: index(20000), mode
      real(kind=8) :: rnorm
      common /nnlscall/ modellib, mda, m, n, nb, x, rnorm, w, zz, index, mode
      common /nnlsdata/ flux, lambda

      real(kind=8) :: grid_copy(m,nb), b(m)
      integer :: i,j

      external :: nnls


!     Make a copy of the grids, but now with the new value for E(B-V).
!     m is the number of wavelength points.
      do i=1, m
         b(i) = flux(i)
         do j=1, nb
            grid_copy(i,j) = modellib(i,j)*exp(-ebv*(lambda(i)/5500.0d0)**(-0.7d0))
         enddo
      enddo

!      write(*,*) 'ok. I am here now'
!      do i=1, 10
!         do j=1, 10
!            write(*,*) grid_copy(i,j)
!         enddo
!      enddo

      call nnls(grid_copy, mda, m, nb, b, x, rnorm, w, zz, index, mode)

!
!     Should really minimize chisq here..
!
!      write(*,*) 'Current parameters=',(x(i),i=1,nb)
      do_continuum_fit = rnorm

      end function


SUBROUTINE fit_continuum1(l, f, mlib, params, mean, sigma, nb_in, m_in, nm, n_in)
    implicit none
    integer, intent(in) :: nb_in, m_in, nm, n_in
    real(kind=8) :: l(m_in), f(m_in), mlib(nm), params(n_in)
    real(kind=8), intent(inout) :: mean, sigma

    real(kind=8) :: lambda(20000), flux(20000), dflux(20000)
    real(kind=8) :: modellib(20000, 100), x(20000), zz(200000), w(20000)
    integer :: index(20000), mode
    real(kind=8) :: rnorm, ax, bx, cx, fa, fb, fc, tol, ebvmin
    real(kind=8) :: do_continuum_fit
    real(kind=8) :: dbrent, minval
    real(kind=8) :: chi2(m_in), best_cont

    integer :: i,j, mda, n, m, nb
    logical :: keep_inside
    external :: do_continuum_fit, dbrent, mnbrak

    common /nnlscall/ modellib, mda, m, n, nb, x, rnorm, w, zz, index, mode
    common /nnlsdata/ flux, lambda

    !     Notice in this routine we expect that F & the model library has
    !     already been multiplied with the weights before calling.

!    write(*,*) m_in, nm, n_in, nb_in
!    write(*,*) mean, sigma

    m = m_in
    if (m_in .gt. 20000) then
        write(6,*) 'WARNING: YOU MUST INCREASE THE NUMBER OF '
        write(6,*) ' WAVELENGHT POINTS IN FIT_CONT_NNLS.F!!!!'
        stop
    endif
    n = n_in
    nb = nb_in
    mda = m

    do i=1, m
        lambda(i) = l(i)
        flux(i) = f(i)
        zz(i) = 0.0d0
        do j=1, nb
            modellib(i,j)=mlib(i + (j-1)*m)
        enddo
    enddo
!    do i=1, 10
!        do j=1, 10
!            write(*,*) modellib(i,j)
!        enddo
!    enddo

    rnorm =0.0d0
    mode = 0
    do j=1, n
        x(j) = 0.0d0
        w(j) = 0.0d0
    enddo

!!  First bracket the minimum
!
!   The variable is E(B-V) and we will let the interval be
!   -1 to 6. Somewhere inside there there ought to be a minimum
    ax = -1.0d0
    bx = 6.0d0
!   cx will be returned from the function, as will fa,fb,fc.
    cx = 0.0d0
    keep_inside=.true.
    call mnbrak(ax,bx,cx,fa,fb,fc,do_continuum_fit, keep_inside)

!    write(*,*) ax, bx,cx,fa,fb,fc   ! MLPG: uncomment for debugging
    if (keep_inside .eqv. .false.) then
        write(6,*) 'Warning:: Continuum fitting failed'
        do i=1, n
            if (i .eq. 1) then
               params(i) = -99.9
            else
               params(i) = -99.9
            endif
        enddo
        mean=-99.9
        sigma=-99.9
        return
    endif

!   Now call the minimization routine.
    tol = 1.0d-7
!   write(*,*) 'Calling dbrent!'
    minval = dbrent(ax,bx,cx,do_continuum_fit,tol,ebvmin)
!
!   Finally use the NNLS parameters from the common block to
!   create the output parameters as well as the best-fit continuum
!   and calculate the best-fit mean chi^2 using meanclip
!
      do i=1, n
         if (i .eq. 1) then
            params(i) = ebvmin
         else
            params(i) = x(i-1)
         endif
      enddo

!      do i=1, m
!         best_cont =0.0
!         do j=1, nb
!            best_cont = best_cont + mlib(i+(j-1)*m)*x(j)
!            if (i .lt. 50) write(*,*) 'Adding x, mlib', j, x(j), mlib(i+(j-1)*m)
!         enddo
!        Notice that both flux & the modellibrary have been
!        normalized by the error in the pixel so this is in fact
!        chi2...
!         chi2(i) = (flux(i) - best_cont)**2
!      enddo

!     At present we use this - mean chi^2 per pixel
    mean = rnorm*rnorm/m
    sigma = -1.0
!   call meanclip(chi2, m, mean, sigma)

!    write(*,*) 'MEAN, SIGMA=', mean, sigma





    end SUBROUTINE

