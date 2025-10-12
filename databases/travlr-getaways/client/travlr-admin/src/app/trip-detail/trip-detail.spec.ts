import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TripDetail } from './trip-detail';

describe('TripDetail', () => {
  let component: TripDetail;
  let fixture: ComponentFixture<TripDetail>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TripDetail]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TripDetail);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
