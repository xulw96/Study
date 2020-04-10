$(function () {
    // game rule click
    $('.rules').click(function () {
        $('.rules').stop().fadeIn(100);
    });
    // close button click
    $('.close').click(function () {
        $('.rules').stop().fadeOut(100);
    });
    // start button click
    $('.start').click(function () {
        $(this).stop().fadeOut(100);
        // progress bar
        progressHandler();
        // wolf show-up
        startWolfAnimation();
    });
    // restart button click
    $('.restart').click(function () {
        $('.mask').stop().fadeOut(100);
        progressHandler();
        startWolfAnimation();
    });

    function progressHandler() {
        // reset progress bar
        $('.progress').css({
            width: 180,
        });
        let timer = setInterval(function () {
           // reduce progress bar
           let progressWidth = $('.progress').width();
           progressWidth -= 1;
           $('.progress').css({
               width: progressWidth
           });
           // check whether bar ends
           if(progressWidth <= 0){
               clearInterval(timer);
               $('.mask').stop().fadeIn(100);
               stopWolfAnimation();
           }
       }, 100);
    }

    let wolfTimer;
    function startWolfAnimation() {
        // arrays of images
        let wolf_1 = ['./images/h0.jpp', './images/h1.jpg'];
        let wolf_2 = ['./images/x0.jpg', './images/x1.jpg'];
        // array of location
        let arrPos = [{left: '100px', top: '115px'}, {left: '20px', top: '160px'}];
        let $wolfImage = $('<img src="" class="wolfImage">');
        // random location
        let posIndex = Math.round(Math.random() * 8);
        $wolfImage.css({
            position: 'absolute',
            left: arrPos[posIndex].left,
            top: arrPos[posIndex].top,
        });
        // random image
        let wolfType = Math.round(Math.random()) == 0? wolf_1 : wolf_2;
        window.wolfIndex = 0;
        window.wolfIndexEnd = 5;
        wolfTimer = setInterval(function () {
            if(wolfIndex > wolfIndexEnd){
                $wolfImage.remove();
                clearInterval(wolfTimer);
            }
            $wolfImage.attr('src', wolfType[wolfIndex]);
            wolfIndex++;
        }, 300);
        $('.container').append($wolfImage);
        gameRules($wolfImage);
    }

    function gameRules($wolfImage) {
        $wolfImage.one('click', function () {
            window.wolfIndex = 5;
            window.wolfIndexEnd = 9;
            let $src = $(this).attr('src');
            let flag = $src.indexOf('h') >= 0;
            if(flag){
                $('.score').text(parseInt($('.score').text()) + 10);
            }else{
                $('.score').text(parseInt($('.score').text()) - 10);
            }
        })
    }

    function stopWolfAnimation() {
        $('.wolfImage').remove();
        clearInterval(wolfTimer);
    }
});