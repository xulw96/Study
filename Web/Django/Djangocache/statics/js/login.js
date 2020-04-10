$(function() {
    $('img').click(function(){
        console.log('click me');
        $(this).attr('src', '/app/getcode/');
    })
});