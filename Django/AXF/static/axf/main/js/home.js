$(function(){
    initTopSwiper();
    initSwiperMenu();
})

function initTopSwiper() {
    let swiper = new Swiper('#topswiper', {
        loop: true,
        autoplay: 3000,
        pagination: '.swiper-pagination'
    })
}

function initSwiperMenu() {
    let swiper = new Swiper('#swiperMenu', {
        slidePerView: 3,
    })
}